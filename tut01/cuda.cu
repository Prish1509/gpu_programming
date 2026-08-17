#include <iostream>
#include <cuda_runtime.h>

int getCoresPerSM(int major, int minor) {
    if (major == 1) return 8;
    if (major == 2) return (minor == 0) ? 32 : 48;
    if (major == 3) return 192;
    if (major == 5) return 128;
    if (major == 6) return (minor == 0) ? 64 : 128;
    if (major == 7) return 64;
    if (major == 8) return (minor == 0) ? 64 : 128;
    if (major == 9) return 128;
    return 0;
}

int main() {

    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    std::cout << "Number of CUDA Devices: " << deviceCount << "\n\n";

    for (int i = 0; i < deviceCount; ++i) {

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        int coresPerSM = getCoresPerSM(prop.major, prop.minor);
        int totalCores = coresPerSM * prop.multiProcessorCount;

        std::cout << "========================================\n";
        std::cout << "CUDA Device " << i << "\n";
        std::cout << "========================================\n";

        std::cout << "Device Name: " << prop.name << "\n";

        std::cout << "Compute Capability: "
                  << prop.major << "." << prop.minor << "\n";

        std::cout << "Total Global Memory: "
                  << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)
                  << " GB\n";

        std::cout << "Number of Multiprocessors: "
                  << prop.multiProcessorCount << "\n";

        std::cout << "Cores per SM: "
                  << coresPerSM << "\n";

        std::cout << "Total CUDA Cores: "
                  << totalCores << "\n";

        std::cout << "Maximum Threads per Block: "
                  << prop.maxThreadsPerBlock << "\n";

        std::cout << "Shared Memory per Block: "
                  << prop.sharedMemPerBlock / 1024
                  << " KB\n";

        std::cout << "Warp Size: "
                  << prop.warpSize << "\n";

        // Additional properties

        std::cout << "Maximum Threads Dimension: "
                  << prop.maxThreadsDim[0] << " x "
                  << prop.maxThreadsDim[1] << " x "
                  << prop.maxThreadsDim[2] << "\n";

        std::cout << "Maximum Grid Size: "
                  << prop.maxGridSize[0] << " x "
                  << prop.maxGridSize[1] << " x "
                  << prop.maxGridSize[2] << "\n";

        std::cout << "Clock Rate: "
                  << prop.clockRate / 1000.0
                  << " MHz\n";

        std::cout << "Memory Clock Rate: "
                  << prop.memoryClockRate / 1000.0
                  << " MHz\n";

        std::cout << "Memory Bus Width: "
                  << prop.memoryBusWidth
                  << " bits\n";

        std::cout << "L2 Cache Size: "
                  << prop.l2CacheSize / 1024
                  << " KB\n";

        std::cout << "Registers per Block: "
                  << prop.regsPerBlock << "\n";

        std::cout << "Concurrent Kernels: "
                  << (prop.concurrentKernels ? "Yes" : "No")
                  << "\n";

        std::cout << "Unified Addressing: "
                  << (prop.unifiedAddressing ? "Yes" : "No")
                  << "\n";

        std::cout << "========================================\n\n";
    }

    return 0;
}
