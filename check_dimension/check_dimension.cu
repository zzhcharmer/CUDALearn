#include <iostream>
#include <cuda_runtime.h>

__global__ void checkIndexKernel(void)
{
    printf("\
    threadIdx:(%d,%d,%d)\
    blockIdx:(%d,%d,%d)\
    blockDim:(%d,%d,%d)\
    gridDim(%d,%d,%d)\n",
    threadIdx.x,threadIdx.y,threadIdx.z,
    blockIdx.x,blockIdx.y,blockIdx.z,
    blockDim.x,blockDim.y,blockDim.z,
    gridDim.x,gridDim.y,gridDim.z);
}

int main(int argc,char **argv) {
    const unsigned int nElem = 6;
    dim3 block(3);
    dim3 grid((nElem + block.x - 1) / block.x);
    std::cout << "grid.x = " << grid.x << "\tgrid.y = " << grid.y << "\tgrid.z= " << grid.z  << std::endl;
    std::cout << "block.x = " << block.x << "\tblock.y = " << block.y << "\tblock.z= " << block.z  << std::endl;
    checkIndexKernel<<< grid, block >>>();
    cudaDeviceReset();
    return 0;
}