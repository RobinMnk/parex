#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "utils/timer.h"
#include "interface.h"

using dVec = thrust::device_vector<float>;

// Device kernel
__global__ void VecAddKernel(const float* A, const float* B, float* C, int N)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Host-callable wrapper
void LaunchVecAdd(int blocksPerGrid, int threadsPerBlock, const float* d_A, const float* d_B, float* d_C, int N) {
    VecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();  // Optional: wait for kernel to finish
}

int testCuda() {
    int N = 100000000;

    std::cout << "Testing CUDA!! " << std::endl;


    Timer t;
    // Allocate input vectors in host memory
    printf("Allocating host memory");
    t.start();
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C(N);
    printf("\t\t [%f s]\n", t.timeSeconds());

    // Initialize input vectors
    printf("Initializing vectors");
    t.start();
    std::iota(h_A.begin(), h_A.end(), 0.f);
    std::iota(h_B.begin(), h_B.end(), 0.f);
    printf("\t\t [%f s]\n", t.timeSeconds());

    // Allocate vectors in device memory
    printf("Allocating & Copying to device");
    t.start();
    dVec d_A = h_A;
    dVec d_B = h_B;
    dVec d_C(N);
    printf("\t [%f s]\n", t.timeSeconds());

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Invoke kernel
    printf("Parallel Add");
    t.start();
    LaunchVecAdd(blocksPerGrid, threadsPerBlock, d_A.data().get(), d_B.data().get(), d_B.data().get(), N);
    printf("\t\t\t [%f s]\n", t.timeSeconds());

    // Copy result back to host
    printf("Copy result back to host");
    t.start();
    thrust::copy(d_B.begin(), d_B.end(), h_C.begin());
    printf("\t [%f s]\n", t.timeSeconds());

    return 0;
}
