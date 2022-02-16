#include <iostream>
#include <cuda_runtime.h> 

using namespace std;

__global__ void cuda_solve(){
    printf("\nKernel: Block ID: %d",blockIdx.x);
    printf("\nKernel: Block dim: %d",blockDim.x);
    printf("\nKernel: Thread ID: %d",threadIdx.x);
}

void test(){
    cout<<"Using GPU";
    cuda_solve<<<2,2>>>();
}
