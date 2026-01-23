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
    const NodeIx* neighbors;          // size: 2 * numEdges
    const NodeIx* ranges;             // size: numNodes+1
    const EdgeIx* active_degrees;     // size: numNodes
    const NodeIx* labels;

    // Buffers for Updates
    const EdgeIx* edgeDeletionBuffer;
    const NodeUpdate* nodeUpdateBuffer;

    __host__ __device__
    inline void deactivateEdge(EdgeIx idx) const;

    __host__ __device__
    inline void handleActiveDegrees(NodeIx idx) const;
};

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


class GraphManager {

    // Graph
    thrust::device_vector<NodeIx> neighbors;
    thrust::device_vector<NodeIx> ranges;

    // Partition
    thrust::device_vector<EdgeIx> active_degrees;   // active_degree == 0 implies that a node is inactive
    thrust::device_vector<NodeIx> labels;
    thrust::device_vector<EdgeIx> volumes;

    // Update buffers
    thrust::device_vector<EdgeIx> edgeDeletionBuffer;
    thrust::device_vector<NodeUpdate> nodeUpdateBuffer;

public:
    NodeIx n{0};
    EdgeIx m{0};

    NodeIx numClusters{1};

    explicit GraphManager(const Graph& graph) :
        n(graph.numNodes),
        m(graph.numEdges),
        neighbors(graph.edges),
        ranges(graph.ranges),
        active_degrees(graph.numNodes),
//        permutation(thrust::make_counting_iterator<NodeIx>(0), thrust::make_counting_iterator(n)),
        labels(graph.numNodes, 0),
        volumes{2 * graph.numEdges},
        edgeDeletionBuffer(2 * graph.numEdges),
        nodeUpdateBuffer(graph.numNodes)
    {
        thrust::transform(ranges.begin() + 1, ranges.end(), ranges.begin(), active_degrees.begin(), thrust::minus<int>());
        std::cout << "Copied Graph to GPU. \t" << neighbors.size() / 2 << " edges copied" << std::endl;
    }

    [[nodiscard]] DevGraph getView() const {
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

    thrust::device_vector<NodeIx>& getLabels() {
        return labels;
    }

    thrust::device_vector<EdgeIx>& getActiveDegrees() {
        return active_degrees;
    }

    thrust::device_vector<EdgeIx>& getVolumes() {
        return volumes;
    }

    std::vector<EdgeIx> downloadDegrees() {
        std::vector<EdgeIx> degrees(n);
        thrust::copy(active_degrees.begin(), active_degrees.end(), degrees.begin());
        return degrees;
    }

    Graph downloadGraph() {
        Graph g(n, m);
        thrust::copy(neighbors.begin(), neighbors.end(), g.edges.begin());
        thrust::copy(ranges.begin(), ranges.end(), g.ranges.begin());
        return g;
    }
};



#endif //PAREX_DEVGRAPH_H
