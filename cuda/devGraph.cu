//
// Created by robin on 14.01.2026.
//

#include <thrust/device_vector.h>
#include <vector>

using NodeIx = unsigned int;
using EdgeIx = unsigned int;

struct SwapPair {
    EdgeIx i;
    EdgeIx j;
};

struct NodeUpdate {
    NodeIx nix;
    EdgeIx diff;
};

// Device Struct
struct DevGraph {
    const NodeIx numNodes;
    const EdgeIx numEdges;

    // Graph
    NodeIx* neighbors;          // size: numEdges
    NodeIx* ranges;             // size: numNodes+1
    EdgeIx* active_degrees;     // size: numNodes

    // Buffers for Updates
    SwapPair* swapBuffer;
    NodeUpdate* nodeUpdateBuffer;

    __device__
    void handleSwaps(NodeIx idx) const {
        SwapPair pair = swapBuffer[idx];
        NodeIx temp = neighbors[pair.i];
        neighbors[pair.i] = neighbors[pair.j];
        neighbors[pair.j] = temp;
    }

    __device__
    void handleActiveDegrees(NodeIx idx) const {
        NodeUpdate pair = nodeUpdateBuffer[idx];
        active_degrees[pair.nix] -= pair.diff;
    }
};

__global__
void swapKernel(DevGraph gr, NodeIx numSwaps) {
    NodeIx idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < numSwaps) gr.handleSwaps(numSwaps);
}

__global__
void degreeKernel(DevGraph gr, NodeIx numUpdates) {
    NodeIx idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numUpdates) gr.handleActiveDegrees(numUpdates);
}

template<typename T>
inline void copyToDevice(const std::vector<T>& elements, T* devTarget, cudaStream_t stream) {
    cudaMemcpyAsync(
        devTarget, elements.data(), elements.size() * sizeof(T), cudaMemcpyHostToDevice, stream
    );
}

void applyGraphUpdates(DevGraph& gr, const std::vector<SwapPair>& swaps, const std::vector<NodeUpdate>& updates, cudaStream_t stream) {
    int threads = 256;
    if (!swaps.empty()) {
        copyToDevice(swaps, gr.swapBuffer, stream);
        auto elements = static_cast<NodeIx>(swaps.size());
        size_t blocks = (elements + threads - 1) / threads;
        swapKernel<<<blocks, threads, 0, stream>>>(gr, elements);
    }

    if (!updates.empty()) {
        copyToDevice(updates, gr.nodeUpdateBuffer, stream);
        auto elements = static_cast<NodeIx>(updates.size());
        size_t blocks = (elements + threads - 1) / threads;
        degreeKernel<<<blocks, threads, 0, stream>>>(gr, elements);
    }
}

class DevGraphManager {
private:
    NodeIx n;
    EdgeIx m;

    // These own the actual GPU memory
    thrust::device_vector<NodeIx> neighbors;
    thrust::device_vector<NodeIx> ranges;
    thrust::device_vector<EdgeIx> active_degrees;

    // Update buffers
    thrust::device_vector<SwapPair> swapBuffer;
    thrust::device_vector<NodeUpdate> nodeUpdateBuffer;


public:
    DevGraphManager(NodeIx numNodes, EdgeIx numEdges, NodeIx maxUpdateSize) : n(numNodes), m(numEdges) {
        neighbors.resize(m);
        ranges.resize(n + 1);
        active_degrees.resize(n);

        swapBuffer.resize(maxUpdateSize);
        nodeUpdateBuffer.resize(maxUpdateSize);
    }

    DevGraph getDeviceView() {
        return DevGraph {
                n,
                m,
                thrust::raw_pointer_cast(neighbors.data()),
                thrust::raw_pointer_cast(ranges.data()),
                thrust::raw_pointer_cast(active_degrees.data()),
                thrust::raw_pointer_cast(swapBuffer.data()),
                thrust::raw_pointer_cast(nodeUpdateBuffer.data())
        };
    }

    // Helper to upload initial data from CPU
    void uploadData(const std::vector<NodeIx>& h_neighbors, const std::vector<NodeIx>& h_ranges) {
        neighbors = h_neighbors;
        ranges = h_ranges;
        // ... etc
    }
};


