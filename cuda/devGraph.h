//
// Created by robin on 19.01.2026.
//

#ifndef PAREX_DEVGRAPH_H
#define PAREX_DEVGRAPH_H

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>
#include <thrust/scatter.h>

#include "types.h"

//struct DevGraph {
//    const NodeIx numNodes;
//    const EdgeIx numEdges;
//
//    // Graph
//    const NodeIx* neighbors;          // size: 2 * numEdges
//    const NodeIx* ranges;             // size: numNodes+1
//    const EdgeIx* active_degrees;     // size: numNodes
//    const NodeIx* labels;
//
//    // Buffers for Updates
//    const EdgeIx* edgeDeletionBuffer;
//    const NodeUpdate* nodeUpdateBuffer;
//
//    __host__ __device__
//    inline void deactivateEdge(EdgeIx idx) const;
//
//    __host__ __device__
//    inline void handleActiveDegrees(NodeIx idx) const;
//};

//__global__
//void deactivateEdgeKernel(DevGraph gr, EdgeIx numEdgeDeletions) {
//    NodeIx idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if(idx < numEdgeDeletions) gr.deactivateEdge(idx);
//}
//
//__global__
//void degreeKernel(DevGraph gr, NodeIx numUpdates) {
//    NodeIx idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx < numUpdates) gr.handleActiveDegrees(idx);
//}


//__global__
//void updateActiveNodesKernel(
//        NodeIx numNodes,
//        const NodeIx* __restrict__ ranges,
//        const NodeIx* __restrict__ neighbors,
//        const NodeIx* __restrict__ labels,
//        EdgeIx* __restrict__ activeDegrees
//) {
//    NodeIx i = blockIdx.x * blockDim.x + threadIdx.x;
//    if (i >= numNodes) return;
//
//    // early exit for inactive nodes
//    const NodeIx clusterId = __ldg(&labels[i]);
//    if(clusterId == numNodes) {
//        activeDegrees[i] = 0;
//        return;
//    }
//
//    const NodeIx start = __ldg(&ranges[i]);
//    const NodeIx end   = __ldg(&ranges[i+1]);
//
//    EdgeIx degree = 0;
//
//    // The #pragma unroll hint tells the compiler to optimize the loop
//    // for small segments, which are common in many graphs.
//#pragma unroll 4
//    for (NodeIx j = start; j < end; ++j) {
//        const NodeIx neighbor = neighbors[j];
//        degree += (__ldg(&ranges[neighbor]) != clusterId);
//    }
//
//    activeDegrees[i] = degree;
//}


class GraphManager {
    // Graph
    thrust::device_vector<NodeIx> neighbors;
    thrust::device_vector<NodeIx> ranges;

    // Update buffers
//    thrust::device_vector<EdgeIx> edgeDeletionBuffer;
//    thrust::device_vector<NodeUpdate> nodeUpdateBuffer;

//    void updateVolumes();

public:
    NodeIx n{0};
    EdgeIx m{0};

    NodeIx numClusters{1};

    explicit GraphManager(const Graph& graph) :
        n(graph.numNodes),
        m(graph.numEdges),
        neighbors(graph.edges),
        ranges(graph.ranges)
//        edgeDeletionBuffer(2 * graph.numEdges),
//        nodeUpdateBuffer(graph.numNodes)
    {
//        thrust::transform(ranges.begin() + 1, ranges.end(), ranges.begin(), active_degrees.begin(), thrust::minus<int>());

        std::cout << "Copied Graph to GPU. \t" << neighbors.size() / 2 << " edges copied" << std::endl;
    }

//    [[nodiscard]] DevGraph getView() const {
//        return DevGraph{
//            n, m,
//            thrust::raw_pointer_cast(neighbors.data()),
//            thrust::raw_pointer_cast(ranges.data()),
//        };
//    }

//    thrust::device_vector<NodeIx>& getLabels() {
//        return labels;
//    }

    thrust::device_vector<NodeIx>& getRanges() {
        return ranges;
    }

    thrust::device_vector<NodeIx>& getNeighbors() {
        return neighbors;
    }

//    thrust::device_vector<EdgeIx>& getActiveDegrees() {
//        return active_degrees;
//    }
//
//    thrust::device_vector<EdgeIx>& getVolumes() {
//        return volumes;
//    }

//    std::vector<EdgeIx> downloadDegrees() {
//        std::vector<EdgeIx> degrees(n);
//        thrust::copy(active_degrees.begin(), active_degrees.end(), degrees.begin());
//        return degrees;
//    }

    Graph downloadGraph() {
        Graph g(n, m);
        thrust::copy(neighbors.begin(), neighbors.end(), g.edges.begin());
        thrust::copy(ranges.begin(), ranges.end(), g.ranges.begin());
        return g;
    }

    void updateLabels(std::vector<NodeIx>& nodeLabels, NodeIx activeClusters) {
//        labels = nodeLabels;
//        numActiveClusters = activeClusters;
//
//        unsigned int blocksPerGrid = (n + threads - 1) / threads;
//
//        updateActiveNodesKernel<<<blocksPerGrid, threads, 0, nullptr>>>(
//            n,
//            thrust::raw_pointer_cast(ranges.data()),
//            thrust::raw_pointer_cast(neighbors.data()),
//            thrust::raw_pointer_cast(labels.data()),
//            thrust::raw_pointer_cast(active_degrees.data())
//        );
//
//        updateVolumes();
    }
};

//void GraphManager::updateVolumes() {
//    // Output vector
//    volumes.resize(numActiveClusters);
//
//    // --- Make copies for sorting ---
//    thrust::device_vector<NodeIx> labels_sorted = labels;
//    thrust::device_vector<float> degrees_sorted = active_degrees;
//
//    // --- Radix sort by labels ---
//    thrust::sort_by_key(
//            labels_sorted.begin(),
//            labels_sorted.end(),
//            degrees_sorted.begin()
//    );
//
//    // --- Reduce by key ---
//    thrust::device_vector<NodeIx> unique_labels(numActiveClusters);
//    thrust::device_vector<float> summed_degrees(numActiveClusters);
//
//    auto end = thrust::reduce_by_key(
//            labels_sorted.begin(),
//            labels_sorted.end(),
//            degrees_sorted.begin(),
//            unique_labels.begin(),
//            summed_degrees.begin()
//    );
//
//    const size_t num_unique = end.first - unique_labels.begin();
//
//    // --- Scatter results into volumes ---
//    thrust::scatter(
//            summed_degrees.begin(),
//            summed_degrees.begin() + num_unique,
//            unique_labels.begin(),
//            volumes.begin()
//    );
//}





#endif //PAREX_DEVGRAPH_H
