#include <stdio.h>
#include <cuda_runtime.h>

__device__ int counter = 0;

__device__ void customBarrier(int totalThreads)
{
    atomicAdd(&counter, 1);

    __syncthreads();

    if (threadIdx.x == 0) {
        while (counter < totalThreads);
    }

    __syncthreads();
}

__global__ void testBarrier(int *result)
{
    int tid = threadIdx.x;

    result[tid] = tid * 10;

    printf("Thread %d reached barrier\n", tid);

    customBarrier(blockDim.x);

    result[tid] += 100;

    printf("Thread %d passed barrier\n", tid);
}

int main()
{
    const int N = 8;

    int h_result[N];
    int *d_result;

    cudaMalloc(&d_result, N * sizeof(int));

    testBarrier<<<1, N>>>(d_result);

    cudaDeviceSynchronize();

    cudaMemcpy(
        h_result,
        d_result,
        N * sizeof(int),
        cudaMemcpyDeviceToHost
    );

    printf("\nFinal Results:\n");

    for (int i = 0; i < N; i++) {
        printf("Thread %d: %d\n", i, h_result[i]);
    }

    cudaFree(d_result);

    return 0;
}
