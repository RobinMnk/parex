#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include "interface.h"

using dVec = thrust::device_vector<float>;

// Declare the CUDA function
//cudaError_t randomWalkStep(int blocksPerGrid, int threadsPerBlock);

int cudaFunction() {
//    int N = 100000000;


        std::cout << "Hello from CUDA!! " << std::endl;


//    cudaError_t err = randomWalkStep(256, 1024);
//    if (err != cudaSuccess) {
//        std::cerr << "CUDA error during kernel launch: " << cudaGetErrorString(err) << std::endl;
//    }


//    Timer t;
//
//    // Allocate input vectors in host memory
//    printf("Allocating host memory");
//    t.start();
//    std::vector<float> h_A(N);
//    std::vector<float> h_B(N);
//    std::vector<float> h_C(N);
//    printf("\t\t [%f s]\n", t.timeSeconds());
//
//    // Initialize input vectors
//    printf("Initializing vectors");
//    t.start();
//    std::iota(h_A.begin(), h_A.end(), 0.f);
//    std::iota(h_B.begin(), h_B.end(), 0.f);
//    printf("\t\t [%f s]\n", t.timeSeconds());
//
//    // Allocate vectors in device memory
//    printf("Allocating & Copying to device");
//    t.start();
//    dVec d_A = h_A;
//    dVec d_B = h_B;
//    dVec d_C(N);
//    printf("\t [%f s]\n", t.timeSeconds());
//
//    int threadsPerBlock = 256;
//    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
//
//    // Invoke kernel
//    printf("Parallel Add");
//    t.start();
//    LaunchVecAdd(blocksPerGrid, threadsPerBlock, d_A.data().get(), d_B.data().get(), d_B.data().get(), N);
//    printf("\t\t\t [%f s]\n", t.timeSeconds());
//
//    // Copy result back to host
//    printf("Copy result back to host");
//    t.start();
//    thrust::copy(d_B.begin(), d_B.end(), h_C.begin());
//    printf("\t [%f s]\n", t.timeSeconds());
//
//    for(int i = 0; i < 5; i++)
//        printf("%f\n", h_C[i]);

    return 0;
}
