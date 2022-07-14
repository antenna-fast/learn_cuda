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
//#include <helper_functions.h>
//#include <nvrtc_helper.h>

using namespace std;


void gpu_init(const cudaDeviceProp &devProp, const int dev){
    cout << "GPU device " << dev << ": " << devProp.name << std::endl;
    cout << "Total Global Memory : " << devProp.totalGlobalMem / 1024.0 / 1024.0 / 1024.0 << " GB" << endl;
    cout << "Clock Rate : " << devProp.clockRate << endl;
    cout << "multiProcessorCount ：" << devProp.multiProcessorCount << std::endl;
    cout << "sharedMemPerBlock ：" << devProp.sharedMemPerBlock / 1024.0 << "KB" << std::endl;
    cout << "maxThreadsPerBlock ：" << devProp.maxThreadsPerBlock << std::endl;
    cout << "maxThreadsPerMultiProcessor ：" << devProp.maxThreadsPerMultiProcessor << std::endl;
    cout << "maxWrapPerMultiProcessor：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;  // wrap
}


// Kernel definition
// vector add  并行相加
// 使用引用返回
__global__ void VectorAdd(const float *A, const float *B, float *C){
    unsigned i = threadIdx.x;
    C[i] = A[i] + B[i];
}


// 2D Thread Index
// 返回运行时间
__global__ void idx(clock_t *t){

    clock_t start_time = clock();

    const unsigned thread_id_x = blockDim.x * blockIdx.x + threadIdx.x;  // 注意，x是水平方向的！与矩阵索引方式不同
    const unsigned thread_id_y = blockDim.y * blockIdx.y + threadIdx.y;

    printf("Current Thread ID : (%d, %d) \n", thread_id_x, thread_id_y);

    *t = clock() - start_time;
}


// main function
int main(){

    // GPU Property
    int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
    gpu_init(devProp, dev);

    // 线程布局
    dim3 block_arrange(2, 2);  // 此时只launch一个block: blockidx全是0
    dim3 thread_arrange(2, 3);  // 一个线程块内的线程排布 x y 方向
    // 输出2D索引
    clock_t t;
    idx<<<block_arrange, thread_arrange>>>(&t);
    cudaDeviceSynchronize();  //
    cout << "time : " << t << endl;  // 执行了多少周期？
    cout << "time : " << t / (devProp.clockRate * 1000) << "s" << endl;  // 执行了多少周期？


    // 最简单的例子，定义2D block，计算矩阵相加
//    int row = 3;
//    int col = 4;
//    int arr[row][col];
//    for(int i = 0; i<row; i++){
//        for(int j=0; j<col;j++){
//            arr[i][j] = i*col + j;
//            cout << arr[i][j] << endl;
//        }
//    }

    int numElements = 50;
    size_t size = numElements * sizeof(float);
    cout<<"size:"<<size<<endl;

    // Allocate host memory
    auto *h_A = reinterpret_cast<float *>(malloc(size));
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

    // Device Memory
    // Allocate the device output vector
    CUdeviceptr d_A;
    // cuMemAlloc(&d_A, size);
    cudaMalloc((void**)&d_A, size);

    CUdeviceptr d_B;
    cudaMalloc((void**)&d_B, size);

    CUdeviceptr d_C;
    cudaMalloc((void**)&d_C, size);

    // 数据拷贝
//    // Copy the host input vectors A and B in host memory to the device input
//    // vectors in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
//    checkCudaErrors(cuMemcpyHtoD(d_A, h_A, size));
//    checkCudaErrors(cuMemcpyHtoD(d_B, h_B, size));

//    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
//    dim3 cudaBlockSize(threadsPerBlock, 1, 1);
//    dim3 cudaGridSize(blocksPerGrid, 1, 1);

    return 0;
}