//
// Created by robin on 19.11.2025.
//

#ifndef RCUT_EXPANDER_HIERARCHY_H
#define RCUT_EXPANDER_HIERARCHY_H

#include "expander_decomposition.h"

#include "../utils/timer.h"
#include <iostream>

class ExpanderHierarchy {

    Graph* fullGraph;

    struct Level {
        Graph graph;
        Partition partition;
        std::vector<EdgeIx> volumes;        // size: number of clusters; volume of each subtree
        std::vector<EdgeWeight> cutSizes;   // size: number of clusters; value of cut leaving each cluster in the current graph
        std::vector<NodeIx> parents;        // size: numNodes of graph in level below; gives clusterId (= nix of parent) for each nix in the graph below
    };

    // levels[i].graph.numNodes == levels[i].volumes.size() == levels[i].cutSizes.size() == levels[i+1].parents.size()

    std::vector<Level> levels;

    Level contract(Partition& part) const {
        if(levels.empty()) return contract<true>(part);
        else return contract<false>(part);
    }

    template<bool first>
    Level contract(Partition& part) const {
        const Graph& graph = part.getGraph();
        if constexpr (!first) {
            assert(graph.numNodes == levels.back().graph.numNodes);
        }
        std::vector<NodeIx> edges(2 * part.getNumCutEdges());
        std::vector<EdgeIx> range(part.numClusters() + 1);
        std::vector<EdgeIx> volumes(part.numClusters(), 0);
        std::vector<EdgeWeight> cutSizes(part.numClusters(), 0);
        range[0] = 0;

        NodeIx nix = 0;
        // to which clusterId does a nix belong
        std::vector<NodeIx> clusterLookup(graph.numNodes, graph.numNodes + 1);
        for(const Cluster& cluster: part) {
            if constexpr (first) {
                volumes[nix] = cluster.volume;
            }
            for (const ClusterVertex& cv: cluster) {
                clusterLookup[cv.nix] = nix;
                if constexpr (!first) {
                    volumes[nix] += levels.back().volumes[cv.nix];
                }
            }
            nix++;
        }

        EdgeIx eix = 0;
        nix = 0;
        for(const Cluster& cluster: part) {
            for (const ClusterVertex& cv: cluster) {
                // iterate all deleted edges
                for (auto it = cluster.edgeEnd(cv); it != graph.nend(cv.nix); ++it) {
                    NodeIx nb = *it; // neighbor in other cluster
                    assert(clusterLookup[nb] != nix);
                    assert(clusterLookup[nb] != graph.numNodes + 1);
                    edges[eix] = clusterLookup[nb];
                    cutSizes[nix]++;
                    eix++;
                }
            }
            assert(cutSizes[nix] == cluster.volume - cluster.internalVolume);
            nix++;
            range[nix] = eix;
        }

        return {
            Graph(std::move(edges), std::move(range), part.numClusters(), part.getNumCutEdges()),
            std::move(part),
            std::move(volumes),
            std::move(cutSizes),
            std::move(clusterLookup)
        };
    }

public:
    explicit ExpanderHierarchy(Graph* gr) : fullGraph{gr} {}

    void build() {
        Graph* current = fullGraph;
        Config conf{};
        Timer t;

        do {
            t.start();
            Partition ed = expanderDecomposition(*current, conf);

            if(ed.getNumCutEdges() > 0.95 * current->numEdges) {
                conf.targetSparsity *= 0.8;
                WARN("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t Adjusting targetSparsity to " << conf.targetSparsity);
                continue;
            }

            NodeIx numClusters = ed.numClusters();
            EdgeIx cutSize = ed.getNumCutEdges();

            Level lvl = contract(ed);
            levels.push_back(lvl); // warning: may invalidate Partition (contains self-pointers)!
            current = &levels.back().graph;

            auto timeSpent = t.timeSeconds();
            WARN("Level " << levels.size() << " (" << timeSpent << "s)\t\t-> " << numClusters << " clusters with " << cutSize << " crossing edges");
        } while(current->numNodes > 1);


        // a bit ugly
        for(Level& lvl : levels) {
            lvl.partition.updateSelfReferences();
        }
        WARN("Completed Hierarchy");
    }

    Graph* getGraph() {
        return fullGraph;
    }

    NodeIx height() {
        return static_cast<NodeIx>(levels.size());
    }

    const Level& lvl(NodeIx l) {
        return levels[l];
    }

    void subtractVolume(NodeIx l, NodeIx nix, EdgeIx vol) {
        levels[l].volumes[nix] -= vol;
    }
    void addCut(NodeIx l, NodeIx nix, EdgeIx cut) {
        levels[l].cutSizes[nix] += cut;
    }
};


#endif //RCUT_EXPANDER_HIERARCHY_H
