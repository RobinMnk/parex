#include <cuda_runtime.h>
#include <thrust/device_vector.h>


struct DevRW {
    __device__ static unsigned int numNodes, numEdges;
    __device__ static float* node_values;
    __device__ static float* dist;
    static float* old_dist;


    static __global__
    void computeNodeValues(const unsigned int* node_degrees) {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if(i < numNodes) {
            node_values[i] = old_dist[i] / static_cast<float>(node_degrees[i]);
        }
    }

    static __global__
    void iterateWalk(const unsigned int* edges, const unsigned int* sl) {
        unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
        if(i < numEdges && sl[i] > 0) {
            dist[edges[i]] += node_values[i];
        }
    }
};


struct RandomWalk {

    thrust::device_vector<float> dist, old_dist;
    thrust::device_vector<float> node_values;

    RandomWalk(unsigned int n_Nodes, unsigned int n_Edges) {
        cudaMemcpyToSymbol(DevRW::numNodes, &n_Nodes, sizeof(unsigned int));
        cudaMemcpyToSymbol(DevRW::numEdges, &n_Edges, sizeof(unsigned int));
        DevRW::dist = thrust::raw_pointer_cast(dist.data());
        DevRW::old_dist = thrust::raw_pointer_cast(old_dist.data());
        DevRW::node_values = thrust::raw_pointer_cast(node_values.data());
    }
};


// Host-callable wrapper
cudaError_t randomWalkStep(int blocksPerGrid, int threadsPerBlock, const unsigned int* activeNodeDegrees, const unsigned int* edges, const unsigned int* sl) {
    std::swap(RandomWalk::dist, RandomWalk::old_dist);
    DevRW::computeNodeValues<<<blocksPerGrid, threadsPerBlock>>>(activeNodeDegrees);
    DevRW::iterateWalk<<<blocksPerGrid, threadsPerBlock>>>(edges, sl);

    return cudaDeviceSynchronize();
}

