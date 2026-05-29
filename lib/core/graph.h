//
// Created by robin on 13.03.2025.
//

#ifndef RCUT_GRAPH_H
#define RCUT_GRAPH_H

#include "definitions.h"
#include <cstdint>
#include <vector>
#include <algorithm>    // binary_search
#include <limits>       // numeric_limits
#include <unordered_map>
#include <unordered_set>

#define NEIGHBORS(__g, __i) auto __end_it = __g.nend(__i); for(auto _it = __g.nbegin(__i); _it != __end_it; ++_it)
#define NEIGHBORSPTR(__g, __i) auto __end_it = __g->nend(__i); for(auto _it = __g->nbegin(__i); _it != __end_it; ++_it)
#define ITER_EIX_P(__g, __i, __eix) for(EdgeIx __eix = __g->ranges.at(__i); __eix < __g->ranges.at(__i+1); __eix++)
#define ITER_EIX(__g, __i, __eix) for(EdgeIx __eix = __g.ranges.at(__i); __eix < __g.ranges.at(__i+1); __eix++)


struct Edge {
    NodeIx fx{0}, tx{0}; // from index, to index
//    Weight wgt{0};

    Edge() = default;

    Edge(NodeIx fx, NodeIx tx) : fx(fx), tx(tx) {}

    bool operator==(const Edge& other) const {
        return fx == other.fx && tx == other.tx;
    }
};

class Graph {
public:
    std::vector<NodeIx> edges;          // size: 2 * numEdges
    std::vector<EdgeIx> ranges;         // size: numNodes + 1
    std::vector<EdgeWeight> weights;    // size: numNodes
    // std::vector<EdgeIx> map;            // in an undirected graph: maps edge in one direction to its counterpart edge
//    std::vector<EdgeIx> edgeToNode;

    NodeIx numNodes;
    EdgeIx numEdges;

    Graph() : edges{}, ranges{}, weights{}, numNodes{0}, numEdges{0} { }

    Graph(NodeIx n, EdgeIx m) : edges(2 * m), ranges(n + 1), weights(2 * m, 1), numNodes{n}, numEdges{m} { }

    Graph(std::vector<NodeIx>&& ed, std::vector<EdgeIx>&& rg, NodeIx nNodes, EdgeIx nEdges) :
            edges(std::move(ed)), ranges(std::move(rg)), weights(2 * nEdges, 1), numNodes{nNodes}, numEdges{nEdges} {
    }

//    Graph(std::vector<NodeIx>&& ed, std::vector<EdgeIx>&& rg, std::vector<EdgeWeight>&& wgt, NodeIx nNodes, EdgeIx nEdges) :
//            edges(std::move(ed)), ranges(std::move(rg)), weights(std::move(wgt)), map(2*nEdges, 0), numNodes{nNodes}, numEdges{nEdges} {
//    }

//    void populateEdgeLookup() {
//        NodeIx nix = 0;
//        for(EdgeIx eix = 0; eix < 2 * numEdges; eix++) {
//            while(ranges[nix+1] <= eix) nix++;
//            edgeToNode[eix] = nix;
//        }
//    }

    [[nodiscard]] inline std::vector<NodeIx>::iterator nbegin(NodeIx node) {
        return edges.begin() + ranges.at(node);
    }

    [[nodiscard]] inline std::vector<NodeIx>::iterator nend(NodeIx node) {
        return edges.begin() + ranges.at(node + 1);
    }

    [[nodiscard]] inline std::vector<NodeIx>::const_iterator nbegin(NodeIx node) const {
        return edges.begin() + ranges.at(node);
    }

    [[nodiscard]] inline std::vector<NodeIx>::const_iterator nend(NodeIx node) const {
        return edges.begin() + ranges.at(node + 1);
    }

    [[nodiscard]] bool hasNeighbor(NodeIx node, NodeIx nb) {
        return std::binary_search(nbegin(node), nend(node), nb);
    }

    [[nodiscard]] inline EdgeIx startIndex(NodeIx node) {
        return ranges.at(node);
    }

//    [[nodiscard]] inline EdgeIx edgeWeight(NodeIx from, NodeIx to) const {
//        return std::distance(nbegin(node), nend(node));
//    }

    [[nodiscard]] inline EdgeIx degree(NodeIx node) const {
        return static_cast<EdgeIx>(std::distance(nbegin(node), nend(node)));
    }

    bool operator==(const Graph& other) const {
        return edges == other.edges
            && ranges == other.ranges
            && weights == other.weights
            && numNodes == other.numNodes
            && numEdges == other.numEdges;
    }
};

class DynamicGraph {
public:
    std::vector<Edge> edges{};
    NodeIx numNodes{0};
    EdgeIx numEdges{0};

    DynamicGraph() = default;

    DynamicGraph(std::vector<Edge>&& eg, NodeIx n, EdgeIx m) : edges(std::move(eg)), numNodes(n), numEdges(m) {}

    void addEdge(NodeIx fx, NodeIx tx) {
        edges.emplace_back(std::min(fx, tx), std::max(fx, tx));
        numEdges++;
    }

    void removeEdge(NodeIx fx, NodeIx tx) {
        if(fx > tx) std::swap(fx, tx);
        auto eit = std::find(edges.begin(), edges.end(), Edge{fx, tx});
        std::swap(*eit, edges.back());
        popEdge();
    }

    void popEdge() {
        edges.pop_back();
        numEdges--;
    }

    void removeEdgeByIx(EdgeIx eix) {
        std::swap(edges.at(eix), edges.back());
        popEdge();
    }

    NodeIx addNode() {
        return numNodes++;
    }

    /// Convert this DynamicGraph instance to a Graph
    [[nodiscard]] Graph finalize() const;
};

inline Graph DynamicGraph::finalize() const {
    if(edges.empty()) return {};
    std::vector<Edge> fullEdges(2 * numEdges);
    std::vector<EdgeIx> rg(numNodes + 1);
    // std::vector<EdgeIx> map(2 * numEdges);

    EdgeIx eix = 0;
    for(const auto& [fx, tx]: edges) {
        fullEdges.at(eix++) = {fx, tx};
        fullEdges.at(eix++) = {tx, fx};
    }

    std::sort(fullEdges.begin(), fullEdges.end(), [](const Edge& a, const Edge& b){
        return a.fx == b.fx ? a.tx < b.tx : a.fx < b.fx;
    });

    eix = 0;
    NodeIx nix = 0;
    std::vector<NodeIx> eg(2 * numEdges);
    while(eix < fullEdges.size()) {
        assert(nix <= numNodes);
        rg.at(nix) = eix;
        while(eix < fullEdges.size() && fullEdges.at(eix).fx == nix) eix++;
        nix++;
    }
    rg.at(numNodes) = eix;

    std::vector<EdgeIx> tmp(rg);
    for(eix = 0; eix < 2 * numEdges; eix++) {
        Edge e = fullEdges.at(eix);
        // EdgeIx pix = tmp.at(e.tx)++;
        // map.at(eix) = pix;
        // map.at(pix) = eix;
        eg.at(eix) = e.tx;
    }

    return {std::move(eg), std::move(rg), numNodes, numEdges};
}

#endif //RCUT_GRAPH_H
