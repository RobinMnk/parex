//
// Created by robin on 19.01.2026.
//

#ifndef PAREX_DEVGRAPH_H
#define PAREX_DEVGRAPH_H

#include "core/definitions.h"

struct DevGraph {
    const NodeIx numNodes;
    const EdgeIx numEdges;

    // Graph
    NodeIx* neighbors;          // size: 2 * numEdges
    NodeIx* ranges;             // size: numNodes+1
    EdgeIx* active_degrees;     // size: numNodes
    NodeIx* labels;

    // Buffers for Updates
    EdgeIx* edgeDeletionBuffer;
    NodeUpdate* nodeUpdateBuffer;

    __host__ __device__
    inline void deactivateEdge(EdgeIx idx) const;

    __host__ __device__
    inline void handleActiveDegrees(NodeIx idx) const;
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


class GraphManager {
    NodeIx n{0};
    EdgeIx m{0};

    // Graph and Partition
    thrust::device_vector<NodeIx> neighbors;
    thrust::device_vector<NodeIx> ranges;
    thrust::device_vector<EdgeIx> active_degrees;
    thrust::device_vector<NodeIx> labels;

    // Update buffers
    thrust::device_vector<EdgeIx> edgeDeletionBuffer;
    thrust::device_vector<NodeUpdate> nodeUpdateBuffer;

public:
    explicit GraphManager(const Graph& graph) :
        n(graph.numNodes), m(graph.numEdges), neighbors(graph.edges), ranges(graph.ranges), active_degrees(graph.numNodes),
        labels(graph.numNodes, 0), edgeDeletionBuffer(2 * graph.numEdges), nodeUpdateBuffer(graph.numNodes)
    {
        thrust::transform(ranges.begin() + 1, ranges.end(), ranges.begin(), active_degrees.begin(), thrust::minus<int>());
        std::cout << "Copied Graph to GPU. \t" << neighbors.size() / 2 << " edges copied" << std::endl;
    }

    DevGraph getView() {
        return DevGraph{
                n, m,
                thrust::raw_pointer_cast(neighbors.data()),
                thrust::raw_pointer_cast(ranges.data()),
                thrust::raw_pointer_cast(active_degrees.data()),
                thrust::raw_pointer_cast(labels.data()),
                thrust::raw_pointer_cast(edgeDeletionBuffer.data()),
                thrust::raw_pointer_cast(nodeUpdateBuffer.data())
        };
    }

    Graph downloadGraph() {
        Graph g(n, m);
        thrust::copy(neighbors.begin(), neighbors.end(), g.edges.begin());
        thrust::copy(ranges.begin(), ranges.end(), g.ranges.begin());
        return g;
    }

};



#endif //PAREX_DEVGRAPH_H
