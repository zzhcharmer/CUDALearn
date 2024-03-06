#include <iostream>
#include <cuda_runtime.h>

__global__ void hello_world(void) {
    printf("GPU: Hello world!\n");
}

int main(int argc,char **argv) {
    std::cout << "CPU: Hello world!" << std::endl;
    hello_world<<<1, 10>>>();
    cudaDeviceReset(); //Destroy all allocations and reset all state on the current device in the current process.
    return 0;
}
