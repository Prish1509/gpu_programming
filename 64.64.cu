#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;
using namespace wmma;



constexpr int TILE = 16;
constexpr int NUM_TILES = 64 / TILE;   
constexpr int MATRIX_SIZE = 64;




__global__ void init(half *A, half *B, int M, int N, int K) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalA = M * K;
    int totalB = K * N;

    if (tid < totalA) {
        A[tid] = __float2half((float)tid);
    }
    if (tid < totalB) {
        B[tid] = __float2half((float)tid);
    }
}









__global__ void tensorCoreGemmKernel(const half *A,
                                     const half *B,
                                     float *C,
                                     int M, int N, int K) {
    
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = globalThreadId / warpSize;

    
    int totalOutputTiles = (M / TILE) * (N / TILE);
    if (warpId >= totalOutputTiles) return;

    
    
    
    int tileRow = warpId / (N / TILE);
    int tileCol = warpId % (N / TILE);

    
    int cRow = tileRow * TILE;
    int cCol = tileCol * TILE;

    
    fragment<matrix_a, TILE, TILE, TILE, half, row_major> aFrag;
    fragment<matrix_b, TILE, TILE, TILE, half, row_major> bFrag;
    fragment<accumulator, TILE, TILE, TILE, float> cFrag;

    
    fill_fragment(cFrag, 0.0f);

    
    
    
    
    
    
    
    for (int tileK = 0; tileK < K / TILE; ++tileK) {
        int aRow = cRow;
        int aCol = tileK * TILE;

        int bRow = tileK * TILE;
        int bCol = cCol;

        
        const half *aTile = A + aRow * K + aCol;
        const half *bTile = B + bRow * N + bCol;

        load_matrix_sync(aFrag, aTile, K);
        load_matrix_sync(bFrag, bTile, N);

        
        
        mma_sync(cFrag, aFrag, bFrag, cFrag);
    }

    
    float *cTile = C + cRow * N + cCol;
    store_matrix_sync(cTile, cFrag, N, mem_row_major);
}


void cpuGemm(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    
    const int M = 64;
    const int N = 64;
    const int K = 64;

    half *devA = nullptr;
    half *devB = nullptr;
    float *devC = nullptr;

    float *hostC = (float *)malloc(M * N * sizeof(float));
    float *referenceC = (float *)malloc(M * N * sizeof(float));
    float *hostA = (float *)malloc(M * K * sizeof(float));
    float *hostB = (float *)malloc(K * N * sizeof(float));

    if (!hostC || !referenceC || !hostA || !hostB) {
        printf("Host allocation failed\n");
        return 1;
    }

    cudaMalloc(&devA, M * K * sizeof(half));
    cudaMalloc(&devB, K * N * sizeof(half));
    cudaMalloc(&devC, M * N * sizeof(float));

    
    init<<<4, 256>>>(devA, devB, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("init launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    tensorCoreGemmKernel<<<1, 512>>>(devA, devB, devC, M, N, K);
    cudaEventRecord(stop);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Tensor Core kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaEventSynchronize(stop);

    float elapsedTime = 0.0f;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(hostC, devC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\n========================================\n");
    printf("64x64 Tensor Core Matrix Multiplication\n");
    printf("========================================\n");
    printf("Matrix size : %dx%d x %dx%d\n", M, K, K, N);
    printf("Tensor tile : %dx%dx%d\n", TILE, TILE, TILE);
    printf("Output tiles: %d x %d = %d\n", NUM_TILES, NUM_TILES,
           NUM_TILES * NUM_TILES);
    printf("Warps used  : %d\n", NUM_TILES * NUM_TILES);
    printf("Threads     : %d\n", NUM_TILES * NUM_TILES * warpSize);
    printf("Kernel time : %f ms\n", elapsedTime);

    
    for (int i = 0; i < M * K; ++i) hostA[i] = (float)i;
    for (int i = 0; i < K * N; ++i) hostB[i] = (float)i;

    cpuGemm(hostA, hostB, referenceC, M, N, K);

    
    int errors = 0;
    float maxError = 0.0f;

    for (int i = 0; i < M * N; ++i) {
        float error = fabsf(hostC[i] - referenceC[i]);
        if (error > maxError) maxError = error;

        
        if (error > 1.0e-2f * fmaxf(1.0f, fabsf(referenceC[i]))) {
            if (errors < 10) {
                printf("Mismatch at index %d: GPU=%f CPU=%f\n",
                       i, hostC[i], referenceC[i]);
            }
            ++errors;
        }
    }

    printf("Max absolute error: %f\n", maxError);

    if (errors == 0)
        printf("RESULT: PASS - Tensor Core result matches CPU result.\n");
    else
        printf("RESULT: FAIL - %d elements differ.\n", errors);

    
    printf("\nFirst 4x4 elements of C:\n");
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("%10.1f ", hostC[i * N + j]);
        }
        printf("\n");
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    free(hostC);
    free(referenceC);
    free(hostA);
    free(hostB);

    return errors == 0 ? 0 : 1;
}
