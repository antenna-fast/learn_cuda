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

using namespace std;


/**
 * 返回运行时间
 * **/
__global__ void idx(clock_t *t){

    clock_t start_time = clock();

    /**
     * insert your code here
     * **/

    *t = clock() - start_time;
}

/**
 * Init Data
 * 初始化Matrix
 * **/
template <typename T>
void InitMat(T *A, int nxy, int val){
    for (size_t i = 0; i < nxy; i++){ 
        A[i] = i;
    }
    return;
}


// 打印矩阵
// 线性存储，行优先
// 矩阵的尺寸： X行 Y列
template <typename T>
void printMat(const T*A, int nx, int ny){
    for (size_t i = 0; i < nx; i++)
    { 
        for (size_t j = 0; j < ny; j++)
        { 
            cout << A[j] << " ";
        }
        cout << endl;
        A += ny;  // 跳转一行的内存地址
    }
    cout << endl;

    return;
}

/**
 * C = A + B on CPU
 * nx: 行数
 * ny: 
 * **/
template <typename T>
void matAdd(const T *A, const T* B, T*C, int nx, int ny){
    // cout << "Matrix size: " << nx << " x " << ny << endl;
    for (size_t i = 0; i < nx; i++)      // 行
    {
        for (size_t j = 0; j < ny; j++)  // 列
        {
            C[j] = A[j] + B[j];
        }
        A += ny;  // 跳转一行的内存地址
        B += ny;
        C += ny;
    }
    return;
}

/**
 * Check the correction of data on CPU
 * nx: row
 * ny: col
 * **/
template <typename T>
bool matCheckEqual(const T *A, const T* B, int nx, int ny){
    // cout << "Matrix size: " << nx << " x " << ny << endl;
    for (size_t i = 0; i < nx; i++)      // per row
    {
        for (size_t j = 0; j < ny; j++)  // per col
        {
            if(A[j] - B[j] > 1e-5){
                return false;
            }
        }
        A += ny;  // 跳转一行
        B += ny;
    }
    return true;
}


/**
 * C = A + B on GPU
 * 2D Thread Index 
 * nx: 行数
 * ny: 列数
 * 如果使用线性存储，其实是又转换成了1D的索引
 * **/
__global__ void matAddCUDA(const float *A, const float *B, float*C, int nx, int ny){
    // calculate 2d idx
    // x是水平方向的！与矩阵索引方式不同
    int thread_id_x = blockDim.x * blockIdx.x + threadIdx.x;  // 行方向
    int thread_id_y = blockDim.y * blockIdx.y + threadIdx.y;  // 列方向

    // Mapping the 2D thread index into 1D index 
    int idx = thread_id_y * ny + thread_id_x;  
    // printf("Current Thread ID : (%d, %d) \n", thread_id_x, thread_id_y);

    if(thread_id_x < ny && thread_id_y < nx){
        C[idx] = A[idx] + B[idx];
    }

}


// main function
int main(){

    bool is_debug = false;
    // bool is_debug = true;

    // Matrix Addition
    // Generate data
    int row = 100;
    int col = 500;
    int numElements = row * col;
    int numBytes = numElements * sizeof(float);
    cout << "Matrix size: row=" << row << " col=" << col << endl;

    // Malloc Host memory
    float *h_A = (float *) malloc(numBytes);
    float *h_B = (float *) malloc(numBytes);
    float *h_C = (float *) malloc(numBytes);
    float *h_CC = (float *) malloc(numBytes);

    // Verify that allocations succeeded
    if (h_A == nullptr || h_B == nullptr || h_C == nullptr) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Init data
    InitMat(h_A, numElements, 1);
    if(is_debug){
        cout << "Matrix A: " << endl;
        printMat(h_A, row, col);
    }

    InitMat(h_B, numElements, 2);
    if(is_debug){
        cout << "Matrix B: " << endl;
        printMat(h_B, row, col);
    }

    // Allocate the device output vector
    float *d_A;
    cudaMalloc((void**)&d_A, numBytes);

    float *d_B;
    cudaMalloc((void**)&d_B, numBytes);

    float *d_C;
    cudaMalloc((void**)&d_C, numBytes);

    // MemCopy: Host to Device
    printf("Copy input data from the host memory to the CUDA device\n");
    // target, source
    cudaMemcpy(d_A, h_A, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, numBytes, cudaMemcpyHostToDevice);

    // ThreadsPerBlock
    // 2D block with 2D thread
    dim3 threadsPerBlock(32, 32);
    cout << "threadDimX: " << threadsPerBlock.x << endl;  // 水平分布的thread
    cout << "threadDimY: " << threadsPerBlock.y << endl;  // 竖直分布的thread
    // cout << "z: " << threadsPerBlock.z << endl;  // default=1

    // BlocksPerGrid
    // 保证xy方向的索引都被检索到
    int blockDimX = (row - 1) / threadsPerBlock.x + 1;
    int blockDimY = (col - 1) / threadsPerBlock.y + 1;
    dim3 blocksPerGrid(blockDimX, blockDimY);
    cout << "blockDimX: " << blocksPerGrid.x << endl;  // 水平分布的block
    cout << "blockDimY: " << blocksPerGrid.y << endl;  // 竖直分布的block

    // Launch the matAdd CUDA Kernel
    matAddCUDA<<<threadsPerBlock, blocksPerGrid>>>(d_A, d_B, d_C, row, col);
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess){
        cout << "ERROR: " << cudaGetErrorString(cudaStatus) << endl;
    }

    // Copy Device to Host
    cudaMemcpy(h_C, d_C, numBytes, cudaMemcpyDeviceToHost);

    // Check the result
    if(is_debug){
        cout << "\nMatrix C: " << endl;
        printMat(h_C, row, col);
    }

    // CPU
    matAdd(h_A, h_B, h_CC, row, col);
    
    bool checkStatus = matCheckEqual(h_C, h_CC, row, col);
    cout << "checkStatus: " << checkStatus << endl;

    return 0;
}