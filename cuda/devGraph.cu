//
// Created by robin on 14.01.2026.
//

#include "interface.h"

#include <thrust/device_vector.h>
#include <vector>

// Device Struct
struct DevGraph {
    const NodeIx numNodes;
    const EdgeIx numEdges;

    // Graph
    NodeIx* neighbors;          // size: 2 * numEdges
    NodeIx* ranges;             // size: numNodes+1
    EdgeIx* active_degrees;     // size: numNodes

    // Buffers for Updates
    EdgeIx* edgeDeletionBuffer;
    NodeUpdate* nodeUpdateBuffer;

    __host__ __device__
    inline void deactivateEdge(EdgeIx idx) const {
        // to deactivate an edge we "redirect it" to point to a garbage-node with index 2 * numEdges
        neighbors[idx] = 2 * numEdges;
    }

    __host__ __device__
    inline void handleActiveDegrees(NodeIx idx) const {
        NodeUpdate pair = nodeUpdateBuffer[idx];
        active_degrees[pair.nix] -= pair.diff;
    }
};

__global__
void deactivateEdgeKernel(DevGraph gr, EdgeIx numEdgeDeletions) {
    NodeIx idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < numEdgeDeletions) gr.deactivateEdge(idx);
}

__global__
void degreeKernel(DevGraph gr, NodeIx numUpdates) {
    NodeIx idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numUpdates) gr.handleActiveDegrees(idx);
}

template<typename T>
inline void copyToDevice(const std::vector<T>& elements, T* devTarget, cudaStream_t stream) {
    cudaMemcpyAsync(
        devTarget, elements.data(), elements.size() * sizeof(T), cudaMemcpyHostToDevice, stream
    );
}

class CudaDeviceManager::Impl {
private:
    NodeIx n{0};
    EdgeIx m{0};

    // Graph and Partition
    thrust::device_vector<NodeIx> neighbors;
    thrust::device_vector<NodeIx> ranges;
    thrust::device_vector<EdgeIx> active_degrees;

    // Update buffers
    thrust::device_vector<EdgeIx> edgeDeletionBuffer;
    thrust::device_vector<NodeUpdate> nodeUpdateBuffer;

public:
    DevGraph getDeviceView() {
        return DevGraph {
                n,
                m,
                thrust::raw_pointer_cast(neighbors.data()),
                thrust::raw_pointer_cast(ranges.data()),
                thrust::raw_pointer_cast(active_degrees.data()),
                thrust::raw_pointer_cast(edgeDeletionBuffer.data()),
                thrust::raw_pointer_cast(nodeUpdateBuffer.data())
        };
    }

    void uploadGraph(const Graph& graph) {
        n = graph.numNodes;
        m = graph.numEdges;
        neighbors = graph.edges;
        ranges = graph.ranges;
        active_degrees.resize(n);
        edgeDeletionBuffer.resize(m);
        nodeUpdateBuffer.resize(n);

        std::cout << "Copied Graph to GPU. \t" << neighbors.size() / 2 << " edges copied" << std::endl;
    }

    void applyGraphUpdates(const std::vector<EdgeIx>& edgeDeletions, const std::vector<NodeUpdate>& updates) {
        DevGraph device = getDeviceView();
        int threads = 256;
        if (!edgeDeletions.empty()) {
            copyToDevice(edgeDeletions, device.edgeDeletionBuffer, nullptr);
            auto elements = static_cast<NodeIx>(edgeDeletions.size());
            size_t blocks = (elements + threads - 1) / threads;
            deactivateEdgeKernel<<<blocks, threads, 0, nullptr>>>(device, elements);

            // Check if the launch itself was valid
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
            }
        }

        if (!updates.empty()) {
            copyToDevice(updates, device.nodeUpdateBuffer, nullptr);
            auto elements = static_cast<NodeIx>(updates.size());
            size_t blocks = (elements + threads - 1) / threads;
            degreeKernel<<<blocks, threads, 0, nullptr>>>(device, elements);
        }

        cudaError_t err = cudaStreamSynchronize(nullptr);
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU Error: %s\n", cudaGetErrorString(err));
        }
    }

    Graph downloadGraph() {
        Graph g(n, m);
        thrust::copy(neighbors.begin(), neighbors.end(), g.edges.begin());
        thrust::copy(ranges.begin(), ranges.end(), g.ranges.begin());
        return g;
    }
};

CudaDeviceManager::CudaDeviceManager() : impl(std::make_unique<Impl>()) {}

CudaDeviceManager::~CudaDeviceManager() = default;

void CudaDeviceManager::uploadGraph(const Graph &graph) { impl->uploadGraph(graph); }

Graph CudaDeviceManager::downloadGraph() { return impl->downloadGraph(); }

void CudaDeviceManager::applyGraphUpdates(const std::vector<EdgeIx>& edgeDeletions, const std::vector<NodeUpdate>& updates) {
    impl->applyGraphUpdates(edgeDeletions, updates);
}



