#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <random>
#include <vector>

struct DevRW {
    DevGraph gr;

    // Data Arrays, size: numNodes.
    float* node_values;         // random walk value of each node (not accumulated)
    float* dist;
    float* old_dist;

    __device__
    void initDistribution(NodeIx i) {
        dist[i] = 0;
    }

    __device__
    void computeNodeValue(NodeIx i) const {
        node_values[i] = old_dist[i] / static_cast<float>(gr.active_degrees[i]);
    }

    __device__
    void iterateWalkStep(unsigned int i, const unsigned int* edges, const unsigned int* sl) {
        if(sl[i] > 0) dist[edges[i]] += node_values[i];
    }
};

__global__
void initDistribution(DevRW rw) {
//    std::normal_distribution<float> normalDist;


}

// Kernel to compute node values
__global__
void computeNodeValuesKernel(DevRW rw) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < rw.gr.numNodes) rw.computeNodeValue(i);
}

// Kernel to perform random walk step
__global__
void iterateWalkKernel(DevRW rw, const unsigned int* edges, const unsigned int* sl) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < rw.numEdges) rw.iterateWalkStep(i, edges, sl);
}

// Host-side wrapper class
struct RandomWalk {
    thrust::device_vector<float> dist;
    thrust::device_vector<float> old_dist;
    thrust::device_vector<float> node_values;
    DevRW rw;

    RandomWalk(unsigned int nNodes, unsigned int nEdges)
            : dist(nNodes, 0.0f),
              old_dist(nNodes, 0.0f),
              node_values(nEdges, 0.0f),
              rw({
                  nNodes,
                  nEdges,
                  thrust::raw_pointer_cast(dist.data()),
                  thrust::raw_pointer_cast(old_dist.data()),
                  thrust::raw_pointer_cast(node_values.data())
              }) {}

    // Launch kernels
    void computeNodeValues(const unsigned int* node_degrees, unsigned int threads = 256) const {
        unsigned int blocks = (rw.numNodes + threads - 1) / threads;
        computeNodeValuesKernel<<<blocks, threads>>>(rw, node_degrees);
        cudaDeviceSynchronize();
    }

    void iterateWalk(const unsigned int* edges, const unsigned int* sl, unsigned int threads = 256) const {
        unsigned int blocks = (rw.numEdges + threads - 1) / threads;
        iterateWalkKernel<<<blocks, threads>>>(rw, edges, sl);
        cudaDeviceSynchronize();
    }

    // Optional: copy back results to host
    [[nodiscard]] thrust::host_vector<float> getDistHost() const {
        return dist;
    }
};

int main() {
    unsigned int nNodes = 10;
    unsigned int nEdges = 15;



    // Example node degrees (device array)
    thrust::device_vector<unsigned int> node_degrees_vec(nNodes, 2); // every node has degree 2
    unsigned int* node_degrees_ptr = thrust::raw_pointer_cast(node_degrees_vec.data());

    // Example edges and sl arrays
    thrust::device_vector<unsigned int> edges_vec(nEdges);
    thrust::device_vector<unsigned int> sl_vec(nEdges, 1); // all valid
    unsigned int* edges_ptr = thrust::raw_pointer_cast(edges_vec.data());
    unsigned int* sl_ptr = thrust::raw_pointer_cast(sl_vec.data());

    RandomWalk rw(nNodes, nEdges);

    // Launch computation
    rw.computeNodeValues(node_degrees_ptr);
    rw.iterateWalk(edges_ptr, sl_ptr);

    // Copy back result to host
    auto result = rw.getDistHost();
    for(auto x : result) std::cout << x << " ";
    std::cout << std::endl;
}
