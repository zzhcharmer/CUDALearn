#include <iostream>
#include <cuda_runtime.h>
const unsigned int ELEMSIZE = 1024;

void gridBlockPrint(unsigned int blockSize) {
    dim3 block(blockSize);
    dim3 grid((ELEMSIZE - 1) /block.x + 1);
    std::cout <<"blockSize = " << blockSize << " grid.x = " << grid.x << " block.x = " << block.x << std::endl;
}

int main(int argc,char ** argv)
{
    gridBlockPrint(1024);
    gridBlockPrint(512);
    gridBlockPrint(256);
    gridBlockPrint(128);

    cudaDeviceReset();
    return 0;
}
