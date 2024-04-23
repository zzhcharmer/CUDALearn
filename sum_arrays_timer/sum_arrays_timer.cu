#include <cuda_runtime.h>
#include <stdio.h>
#include <string>
#include "utils.h"

void sumArrays(const float * a, const float * b, float * res, const int size)
{
    for(unsigned int i = 0; i < size; i ++)
    {
        res[i] = a[i] + b[i];
    }
}

__global__ void sumArraysKernel(const float *a, const float *b, float *res, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        res[i] = a[i] + b[i];
    }
}

__global__ void vectorizedSumArraysKernel(float *a, float *b, float *res, const int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = idx; i < N / 4; i += blockDim.x * gridDim.x) {

		float4 a1 = reinterpret_cast<float4*>(a)[i];
		float4 b1 = reinterpret_cast<float4*>(b)[i];
		float4 c1;
		c1.x = a1.x + b1.x;
		c1.y = a1.y + b1.y;
		c1.z = a1.z + b1.z;
		c1.w = a1.w + b1.w;
		reinterpret_cast<float4*>(res)[i] = c1;
	}
}


void cudaSumArrays(const int nElem, float *a_d, float *b_d, float *res_d,
    float *res_from_gpu_h, float *res_h, std::string funcDesc) {
    // 声明开始和结束事件
    cudaEvent_t start, stop;
    CHECKCUDAERR(cudaEventCreate(&start));
    CHECKCUDAERR(cudaEventCreate(&stop));

    dim3 block(32);
    dim3 grid((nElem + block.x - 1) / block.x);
    // 记录开始事件
    cudaEventRecord(start);
    sumArraysKernel<<<grid, block>>>(a_d, b_d, res_d, nElem);
    cudaEventRecord(stop);

    // 等待kernel执行完成
    cudaEventSynchronize(stop);
    // 计算事件时间差
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    printf("Execution %s<<<%d,%d>>> Time elapsed %f ms\n", funcDesc.c_str(), grid.x, block.x, time);
    CHECKCUDAERR(cudaMemcpy(res_from_gpu_h, res_d, sizeof(float) * nElem, cudaMemcpyDeviceToHost));

    checkResult(res_h, res_from_gpu_h, nElem);
}

void cudaVectorizedSumArrays(const int nElem, float *a_d, float *b_d, float *res_d,
    float *res_from_gpu_h, float *res_h, std::string funcDesc) {
    // 声明开始和结束事件
    cudaEvent_t start, stop;
    CHECKCUDAERR(cudaEventCreate(&start));
    CHECKCUDAERR(cudaEventCreate(&stop));

    int block = 32;
    int grid = (nElem + block - 1) / block;
    // 记录开始事件
    cudaEventRecord(start);
    vectorizedSumArraysKernel<<<grid / 4, block>>>(a_d, b_d, res_d, nElem);
    cudaEventRecord(stop);

    // 等待kernel执行完成
    cudaEventSynchronize(stop);
    // 计算事件时间差
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    printf("Execution %s<<<%d,%d>>> Time elapsed %f ms\n", funcDesc.c_str(), grid, block, time);
    CHECKCUDAERR(cudaMemcpy(res_from_gpu_h, res_d, sizeof(float) * nElem, cudaMemcpyDeviceToHost));

    checkResult(res_h, res_from_gpu_h, nElem);
}

int main(int argc, char **argv)
{
    int dev = 0;
    CHECKCUDAERR(cudaSetDevice(dev));
    
    int nElem = 1 << 14;
    if (cmdUtils::checkCmdLineFlag(argc, const_cast<const char **>(argv), "nums")) {
        nElem = cmdUtils::getCmdLineArgumentInt(argc, const_cast<const char **>(argv), "nums=");
    }

    int useVec = 0;
    if (cmdUtils::checkCmdLineFlag(argc, const_cast<const char **>(argv), "useVec")) {
        useVec = cmdUtils::getCmdLineArgumentInt(argc, const_cast<const char **>(argv), "useVec=");
    }

    printf("Vector size:%d\n",nElem);
    
    int nByte = sizeof(float) * nElem;
    float *a_h = (float*)malloc(nByte);
    float *b_h = (float*)malloc(nByte);
    float *res_h = (float*)malloc(nByte);
    float *res_from_gpu_h = (float*)malloc(nByte);
    memset(res_h, 0, nByte);
    memset(res_from_gpu_h, 0, nByte);

    float *a_d, *b_d, *res_d;
    CHECKCUDAERR(cudaMalloc((float**)&a_d, nByte));
    CHECKCUDAERR(cudaMalloc((float**)&b_d, nByte));
    CHECKCUDAERR(cudaMalloc((float**)&res_d, nByte));

    initialData(a_h, nElem);
    initialData(b_h, nElem);

    CHECKCUDAERR(cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(b_d, b_h, nByte, cudaMemcpyHostToDevice));

    double iStart, iElaps;
    iStart = cpuSecond();
    sumArrays(a_h, b_h, res_h, nElem);
    iElaps = cpuSecond() - iStart;
    printf("Execution CPU Time elapsed %f ms\n", iElaps * 1000.0);

    if (useVec == 1)
    {
        cudaVectorizedSumArrays(nElem, a_d, b_d, res_d, res_from_gpu_h, res_h, "cuda vectorize");
    } else {
        cudaSumArrays(nElem, a_d, b_d, res_d, res_from_gpu_h, res_h, "cuda");
    }

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);

    free(a_h);
    free(b_h);
    free(res_h);
    free(res_from_gpu_h);
    return 0;
}
