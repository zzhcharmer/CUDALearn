#pragma once
#include <ctime>
#include <sys/time.h>
#include <string>

#ifndef STRNCASECMP
#define STRNCASECMP strncasecmp
#endif

#ifdef __DRIVER_TYPES_H__
static const char *_cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorName(error);
}
#endif

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#ifdef __DRIVER_TYPES_H__
#define CHECKCUDAERR(val) check((val), #val, __FILE__, __LINE__)
#endif

namespace cmdUtils {

inline int stringRemoveDelimiter(char delimiter, const char *string) {
    int string_start = 0;

    while (string[string_start] == delimiter) {
        string_start++;
    }

    if (string_start >= static_cast<int>(strlen(string) - 1)) {
        return 0;
    }

    return string_start;
}
// checkCmdLineFlag
inline bool checkCmdLineFlag(const int argc, const char **argv, const char *string_ref) {
    bool bFound = false;
    if (argc >= 1) {
        for (int i = 1; i < argc; i++) {
            int string_start = stringRemoveDelimiter('-', argv[i]);
            const char *string_argv = &argv[i][string_start];

            const char *equal_pos = strchr(string_argv, '=');
            int argv_length = static_cast<int>(
                equal_pos == 0 ? strlen(string_argv) : equal_pos - string_argv);

            int length = static_cast<int>(strlen(string_ref));

            if (length == argv_length && !STRNCASECMP(string_argv, string_ref, length)) {
                bFound = true;
                continue;
            }
        }
    }
    return bFound;
}

inline int getCmdLineArgumentInt(const int argc, const char **argv, const char *string_ref) {
    bool bFound = false;
    int value = -1;

    if (argc >= 1) {
        for (int i = 1; i < argc; i++) {
            int string_start = stringRemoveDelimiter('-', argv[i]);
            const char *string_argv = &argv[i][string_start];
            int length = static_cast<int>(strlen(string_ref));

            if (!STRNCASECMP(string_argv, string_ref, length)) {
                if (length + 1 <= static_cast<int>(strlen(string_argv))) {
                int auto_inc = (string_argv[length] == '=') ? 1 : 0;
                value = atoi(&string_argv[length + auto_inc]);
                } else {
                value = 0;
                }

                bFound = true;
                continue;
            }
        }
    }

    if (bFound) {
        return value;
    }
    
    return 0;
}

}

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);
}

void initialData(float* ip,int size)
{
    time_t t;
    srand((unsigned)time(&t));
    for(unsigned int i = 0; i < size; i++)
    {
        ip[i]=(float)(rand() & 0xffff) / 1000.0f;
    }
}

void initialData_int(int* ip, int size)
{
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i<size; i++)
	{
		ip[i] = int(rand()&0xff);
	}
}
void printMatrix(float * C, const int nx, const int ny)
{
    float *ic = C;
    printf("Matrix<%d,%d>:\n",ny,nx);
    for(int i=0;i<ny;i++)
    {
        for(int j=0;j<nx;j++)
        {
        printf("%6f ",ic[j]);
        }
        ic+=nx;
        printf("\n");
    }
}

void initDevice(int devNum)
{
    int dev = devNum;
    cudaDeviceProp deviceProp;
    CHECKCUDAERR(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using device %d: %s\n", dev, deviceProp.name);
    CHECKCUDAERR(cudaSetDevice(dev));
}

void checkResult(float * hostRef, float * gpuRef, const int N)
{
    double epsilon= 1.0E-8;
    for(int i=0; i<N; i++)
    {
        if(abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
        printf("Results don\'t match!\n");
        printf("%f(hostRef[%d] )!= %f(gpuRef[%d])\n", hostRef[i], i, gpuRef[i], i);
        return;
        }
    }
    printf("Check result success!\n");
}
