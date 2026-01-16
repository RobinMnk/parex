//
// Created by robin on 12.12.2025.
//

#ifndef RCUT_NORMALIZED_CUT_H
#define RCUT_NORMALIZED_CUT_H

#include "expander_hierarchy.h"
#include <set>


class NormalizedCut {

    struct TreeNode {
        NodeIx level{0};
        NodeIx nix{0};
        auto operator<=>(const TreeNode&) const = default;
    };

    struct Candidate {
        TreeNode tn;
        frac_t penalty;
    };

    ExpanderHierarchy* hierarchy;
    Graph* graph;
    std::vector<NodeIx> labels;
    std::set<TreeNode> roots; // for some reason, hashing TreeNode is very annoying ... so using set instead

    [[nodiscard]] const Cluster& children(const TreeNode& tn) const {
        return hierarchy->lvl(tn.level).partition.getCluster(tn.nix);
    }
    [[nodiscard]] EdgeIx volume(const TreeNode& tn) const {
        return hierarchy->lvl(tn.level).volumes[tn.nix];
    }
    [[nodiscard]] EdgeIx cutSize(const TreeNode& tn) const {
        return hierarchy->lvl(tn.level).cutSizes[tn.nix];
    }

    /**
     * Increase to normalized cut objective when cutting this TreeNode, assuming it belongs to a cluster with given cutSize and volume
     */
    [[nodiscard]] frac_t edgePenalty(const TreeNode& tn, EdgeIx clusterCutSize, EdgeIx clusterVolume) const {
        const EdgeWeight cutA = hierarchy->lvl(tn.level).cutSizes[tn.nix];
        const EdgeIx     volA = hierarchy->lvl(tn.level).volumes[tn.nix];

        const EdgeWeight cutB = clusterCutSize + cutA;
        const EdgeIx     volB = clusterVolume - volA;

        return static_cast<frac_t>(cutA) / volA + static_cast<frac_t>(cutB) / volB;
    }

    /**
     * Traverse hierarchy from tn downwards, find the edge minimizing the cutting-penalty, assign each graph node the correct label
     */
    Candidate getCandidate(const TreeNode& tn, NodeIx label, EdgeIx clusterCutSize, EdgeIx clusterVolume) {
        Candidate bestCandidate{TreeNode{0, 0}, 100000};

        for(const ClusterVertex& cv: children(tn)) {
            if(tn.level == 0) {
                labels[cv.nix] = label;
            } else {
                const TreeNode child{tn.level - 1, cv.nix};

                if(roots.contains(child)) continue; // O(log k)

                if(volume(child) < clusterVolume) {
                    frac_t edgePen = edgePenalty(child, clusterCutSize, clusterVolume);
                    if(edgePen < bestCandidate.penalty) {
                        bestCandidate = { child, edgePen };
                    }
                }

                // recursive call in level below
                Candidate cd = getCandidate(child, label, clusterCutSize, clusterVolume);
                if(cd.penalty < bestCandidate.penalty) {
                    bestCandidate = cd;
                }
            }
        }
        return bestCandidate;
    };

    void split(const TreeNode& tn) {
        const EdgeIx volA = volume(tn);
        const EdgeWeight cutA = cutSize(tn);

        NodeIx child = tn.nix;
        NodeIx lvlIdx = tn.level;

        while (lvlIdx + 1 < hierarchy->height()) {
            lvlIdx++;
            NodeIx parent = hierarchy->lvl(lvlIdx).parents[child];
            hierarchy->subtractVolume(lvlIdx, parent, volA);
            hierarchy->addCut(lvlIdx, parent, cutA);

            child = parent;
        }
    }

public:
    explicit NormalizedCut(ExpanderHierarchy* eh) : hierarchy{eh}, graph{eh->getGraph()}, labels(graph->numNodes, 0), roots() {}

    /**
     * Splits the Graph into k clusters, minimizing the normalized cut objective. Adjusts the sparsifier accordingly and returns the resulting partition.
     * @param k number of clusters
     */
    Partition compute(NodeIx k) {
        roots.emplace(hierarchy->height() - 1, 0);

        for(NodeIx i = 1; i <= k; i++) {
            Candidate bestCandidate{TreeNode{0, 0}, 1000};
            NodeIx rootIx = 0;
            frac_t bestDelta = 100000;
            for(const TreeNode& root: roots) {
                EdgeIx rootVol = volume(root);
                EdgeIx rootCutSize = hierarchy->lvl(root.level).cutSizes[root.nix];

                // what would be the best split within the cluster corresponding to tree node "root"?
                Candidate cd = getCandidate(root, rootIx, rootCutSize, rootVol);

                // cd.penalty gives cost of cutting the cheapest edge, rootBias gives contribution cost of splitting this root
                frac_t rootBias = static_cast<frac_t>(rootCutSize) / rootVol;

                // change in normalized cut value: cd.penalty - rootBias
                frac_t delta = cd.penalty - rootBias;

                if(delta < bestDelta) {
                    bestDelta = delta;
                    bestCandidate = cd;
                }

                rootIx++;
            }
            split(bestCandidate.tn);
            roots.insert(bestCandidate.tn);
        }

        Partition part{graph};
        std::vector<NodeIx> mod;
        // this call modifies the underlying graph which invalidates the sparsifier!!
        part.splitByIndices<false, false>(0, labels, k, mod);
        return part;
    }
};


#endif //RCUT_NORMALIZED_CUT_H
