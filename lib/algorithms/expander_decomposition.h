//
// Created by robin on 13.03.2025.
//

#ifndef RCUT_EXPANDER_DECOMPOSITION_H
#define RCUT_EXPANDER_DECOMPOSITION_H

#include "partition.h"
#include "random_walk.h"
#include "cuda/interface.h"
#include <numeric>

struct Strategy {
    enum ClusterResetStrategy   { RECENTER, RESET };
    enum CutStrategy            { CASCADE, ITERATE };

    ClusterResetStrategy    crs{RECENTER};
    CutStrategy             cs{ITERATE};
    bool                    cutWhenConverged{false};
};


// Configuration for Expander Decomposition
struct Config {
    frac_t targetSparsity = 0.2;
    frac_t randomWalkThreshold = 1e-5;
    size_t initialNumSteps = 1; // initial number of rw steps, before the first sweep cut is made
    size_t mainNumSteps = 1;
};


template<Strategy strategy = {}>
Partition expanderDecomposition(Graph& graph, const Config& config = {}) {
    RandomWalk walk{graph.numNodes};
    Partition decomposition{&graph};

    cudaFunction();

    std::vector<NodeIx> active, next(decomposition.numClusters());
    std::iota(next.begin(), next.end(), 0);

    // Graph consists of multiple connected components -> reset random walk in each
    if(decomposition.numClusters() > 1) {
        for(const Cluster& cl: decomposition)
            walk.recenterCluster(cl);
    }

    size_t numSteps = config.initialNumSteps;

    while(!next.empty()) {
        std::swap(active, next);
        next.clear(); // can be replaced by custom queue with lazy delete

        size_t rw = numSteps;
        while (rw--) walk.iterate(decomposition, active);
        auto& values = walk.values();

        while(!active.empty()) {
            NodeIx clusterId = active.back();
            active.pop_back();

            const Cluster& current = decomposition.getCluster(clusterId);

            if(std::abs(walk.averageOf(current)) > config.randomWalkThreshold) {
                WARN("\nCLUSTER CAN NEVER CONVERGE!!!\n");
            }

            LOG("  working on cluster " << clusterId << "\t (" << current.size() << " nodes, potential: " << walk.potential(current) << ")");

            if constexpr (!strategy.cutWhenConverged) {
                if(walk.potential(current) < config.randomWalkThreshold) {
                    LOG("       -> potential sufficiently low -> certified expander");
                    continue;
                }
            }

            auto sweepCut = decomposition.sweepCut(clusterId, values);

            LOG("     Sweep cut:   sparsity=" << sweepCut.sparsity << ", offset=" << sweepCut.offset);

            if(sweepCut.sparsity < config.targetSparsity) {
                LOG("       -> considered sparse -> cutting");
                std::vector<NodeIx> modified{clusterId};
                decomposition.split<false>(clusterId, sweepCut.offset, modified);

                numSteps = config.mainNumSteps;

                for (NodeIx clx: modified) {
                    const Cluster &cl = decomposition.getCluster(clx);
                    if (cl.size() < 2 || cl.internalVolume == 0) continue;
                    frac_t potential;
                    if constexpr (strategy.crs == Strategy::RECENTER)  potential = walk.recenterCluster(cl);
                    if constexpr (strategy.crs == Strategy::RESET)     potential = walk.resetCluster(cl);

                    if constexpr (strategy.cutWhenConverged) {
                        if constexpr (strategy.cs == Strategy::CASCADE) {
                            // cascade sweep cuts by adding cluster clx to active
                            active.push_back(clx);
                        } else if constexpr (strategy.cs == Strategy::ITERATE) {
                            // add clx directly to the next round, sweep cut will be considered after another step of the random walk
                            next.push_back(clx);
                        }
                    } else {
                        // if strategy.cutWhenConverged is set to false, we only consider cluster clx if its potential is above the threshold
                        if (potential > config.randomWalkThreshold) {
                            if constexpr (strategy.cs == Strategy::CASCADE) {
                                active.push_back(clx);
                            } else if constexpr (strategy.cs == Strategy::ITERATE) {
                                next.push_back(clx);
                            }
                        }
                    }
                }
            } else {
                if constexpr (strategy.cutWhenConverged) {
                    if(walk.potential(current) > config.randomWalkThreshold) {
                        next.push_back(clusterId);
                    } else {
                        LOG("       -> potential sufficiently low -> certified expander");
                    }
                } else {
                    next.push_back(clusterId);
                }
            }
        }
    }

    return decomposition;
}

#endif //RCUT_EXPANDER_DECOMPOSITION_H
