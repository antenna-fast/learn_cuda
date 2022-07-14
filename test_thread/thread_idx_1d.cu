//
// Created by yaohua on 2020/6/21.
//

// Cpp sys
#include <iostream>
#include <string>
#include <cstdio>
#include <cmath>
#include <vector>
#include <ctime>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda.h>
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
// #include <helper_functions.h>
#include "helper_cuda.h"
//#include <nvrtc_helper.h>

using namespace std;


// Kernel definition: vector addition: C=A+B
// 使用引用返回
__global__ void VectorAdd(const float *A, const float *B, float *C, int numElements){
    unsigned i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements){
        C[i] = A[i] + B[i];
    }
}


int main(){
    // Generate data
    int numElements = 500000;
    size_t size = numElements * sizeof(float);
    cout<<"size:"<<size<<endl;

    // Allocate host memory
    auto *h_A = reinterpret_cast<float *>(malloc(size));  // 显式强制类型转换
    auto *h_B = reinterpret_cast<float *>(malloc(size));
    auto *h_C = reinterpret_cast<float *>(malloc(size));

    // Verify that allocations succeeded
    if (h_A == nullptr || h_B == nullptr || h_C == nullptr) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / static_cast<float>(RAND_MAX);
        h_B[i] = rand() / static_cast<float>(RAND_MAX);
    }

    // Allocate the device output vector
    cudaError_t err = cudaSuccess;

    // CUdeviceptr d_A;
    float *d_A = NULL;  // 定义在主机端的指针，指向设备端的内存，不可以在主机端解引用
    err = cudaMalloc((void **)&d_A, size);

    // CUdeviceptr d_B;
    float *d_B = NULL;
    cudaMalloc((void**)&d_B, size);

    // CUdeviceptr d_C;
    float *d_C = NULL;
    cudaMalloc((void**)&d_C, size);

    // Copy the host input vectors A and B in host memory to 
    // the device input vectors in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice); 

    // Launch the Vector Add CUDA Kernel
    // int threadsPerBlock = 256;
    int threadsPerBlock = 1024;  // 最大是1024 per block
    // 除法向下取整, 所以要+threadsPerBlock - 1，防止启动的线程数小于数据数
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;  
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    VectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    cudaError_t cudaStatus = cudaGetLastError();  // 检查和函数运行是否正确
    if (cudaStatus != cudaSuccess){
        printf("CUDA Kernel Failed: %s", cudaGetErrorString(cudaStatus));
    }

    // Copy the device result vector in device memory to 
    // the host result vector in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    // (target, source)
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    
    // else
    printf("Test PASSED\n");

    // Free memory
    printf("Free memory\n");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}