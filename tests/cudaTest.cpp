//
// Created by robin on 20.01.2026.
//

#include <gtest/gtest.h>
#include "core/graph.h"
#include "core/partition.h"
#include "algorithms/random_walk.h"
#include "../cuda/interface.h"
#include "utils/graph_io.h"
#include "timer.h"

constexpr auto filename = "../../graphs/mock.mtx";

class CudaTest : public ::testing::TestWithParam<int> {
protected:
    static Graph graph;
    static CudaDeviceManager cuda;

    [[maybe_unused]] static void SetUpTestSuite() {
        graph = readDynGraph(filename).finalize();
        cuda.initialize(graph);
    }

    static void resetGraph() {
        graph = readDynGraph(filename).finalize();
        cuda.initialize(graph);
    }
};
Graph CudaTest::graph{};
CudaDeviceManager CudaTest::cuda{};

TEST_F(CudaTest, UploadGraph) {
    Graph G2 = cuda.downloadGraph();
    ASSERT_EQ(graph, G2);
}

TEST_F(CudaTest, Degrees) {
    auto deg = cuda.downloadDegrees();

    std::vector<EdgeIx> expected(graph.numNodes);
    for(NodeIx nix = 0; nix < graph.numNodes; nix++) {
        expected[nix] = graph.degree(nix);
    }

    ASSERT_EQ(deg, expected);
}

TEST_P(CudaTest, RandomWalkTest) {
    auto rwData = cuda.readRandomWalkValues();

    int numSteps = GetParam();

    RandomWalk rw(graph.numNodes);
    rw.setData(rwData);
    Partition part(&graph);

    for(int i = 0; i < numSteps; i++) {
        rw.iterate(part, {0});
        cuda.iterateRandomWalk();
    }

    auto y = cuda.readRandomWalkValues();
    auto z = rw.values();

    for(NodeIx nix = 0; nix < graph.numNodes; nix++) {
        EXPECT_NEAR(y[nix], z[nix], 0.00001);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Iterations,
    CudaTest,
    // testing::Values(6)
    testing::Values(0, 1, 2, 4, 8, 16, 64, 128, 256, 512)
);


TEST_F(CudaTest, SweepCutTest) {
    auto rwData = cuda.readRandomWalkValues();
    RandomWalk rw(graph.numNodes);
    rw.setData(rwData);
    Partition part(&graph);

    // take 20 steps
    for(int i = 0; i < 20; i++) {
        rw.iterate(part, {0});
        cuda.iterateRandomWalk();
    }

    // confirm active degrees
    auto deg = cuda.downloadDegrees();
    for(const ClusterVertex& cv: part.getCluster(0)) {
        ASSERT_EQ(deg[cv.nix], cv.internalDegree);
    }

    // confirm random walk values
    auto y = cuda.readRandomWalkValues();
    auto z = rw.values();

    for (NodeIx nix = 0; nix < graph.numNodes; nix++) {
        ASSERT_NEAR(y[nix], z[nix], 0.00001) << "nix: " << nix;
    }

    // Sweep Cut
    SweepCut expected = part.sweepCut(0, z);

    cuda.computeSweepCuts();
    AllSweepCuts result = cuda.readSweepCuts();

    // std::vector<NodeData> pt = cuda.downloadPartition();

    NodeIx n = graph.numNodes;

//    for(int j = 0; j < 20; j++) {
//        std::cout << "CPU:  " << expected.pS[j].edgeDiff << " / " << expected.pS[j].vol << " (" << (static_cast<float>(expected.pS[j].edgeDiff) / expected.pS[j].vol) << ")" << "\t\t"
//                << "GPU:  " << pt[j].prefixEdgeDiff << " / " << pt[j].prefixVolume << " (" << (static_cast<float>(pt[j].prefixEdgeDiff) / pt[j].prefixVolume) << ")" << std::endl;
//    }

//     for(int j = 0; j < n; j++) {
// //        EXPECT_EQ(result.prefixSums[j].volume, expected.pS[j].volume);
//         ASSERT_EQ(pt[j].label, 0);
//         ASSERT_EQ(pt[j].nix, j);
//     }

//    std::cout << "cutting at index: " << result.cuts[0].offset << "\n compared to " << expected.offset << std::endl;

//    EXPECT_EQ(expected.pS, result.prefixSums);

    EXPECT_EQ(result.clusterIds.size(), 1);
    EXPECT_EQ(result.cuts.size(), 1);

    EXPECT_EQ(result.clusterIds[0], 0);

//    EXPECT_EQ(result.offsets[0], expected.offset);
    EXPECT_NEAR(result.cuts[0].sparsity, expected.sparsity, 0.0000001);
}



TEST_P(CudaTest, SweepCut) {
    auto rwData = cuda.readRandomWalkValues();
    RandomWalk rw(graph.numNodes);
    rw.setData(rwData);
    Partition part(&graph);

    int numSteps = GetParam();

    for(int i = 0; i < numSteps; i++) {
        rw.iterate(part, {0});
        cuda.iterateRandomWalk();

        SweepCut expected = part.sweepCut(0, rw.values());

        cuda.computeSweepCuts();
        AllSweepCuts result = cuda.readSweepCuts();

        EXPECT_EQ(result.clusterIds.size(), 1);
        EXPECT_EQ(result.cuts.size(), 1);

        EXPECT_EQ(result.clusterIds[0], 0);

//        EXPECT_EQ(result.offsets[0], expected.offset);
        EXPECT_NEAR(result.cuts[0].sparsity, expected.sparsity, 0.00000001);
    }
}

std::vector<NodeIx> getLabels(const Graph& graph, const Partition& part) {
    NodeIx nix = 0;
    // to which clusterId does a nix belong
    std::vector<NodeIx> clusterLookup(graph.numNodes, graph.numNodes + 1);
    for(const Cluster& cluster: part) {
        for (const ClusterVertex& cv: cluster) {
            clusterLookup[cv.nix] = nix;
        }
        nix++;
    }
    return clusterLookup;
}


TEST_F(CudaTest, CutTest) {
    auto rwData = cuda.readRandomWalkValues();
    RandomWalk rw(graph.numNodes);
    rw.setData(rwData);
    Partition part(&graph);

    // take 10 steps
    for(int i = 0; i < 10; i++) {
        rw.iterate(part, {0});
        cuda.iterateRandomWalk();
    }

    auto y = cuda.readRandomWalkValues();
    auto z = rw.values();

    for (NodeIx nix = 0; nix < graph.numNodes; nix++) {
        ASSERT_NEAR(y[nix], z[nix], 0.00001) << "Error for nix = " << nix;
    }

    Timer t;

    t.start();
    // this only cuts cluster 0
    SweepCut sweepCut = part.sweepCut(0, z);
    std::vector<NodeIx> modified{0};
    part.split<false, false>(0, sweepCut.offset, modified);
    auto timeCPU = t.timeMicros();
    std::cout << "CPU time: " << timeCPU << "μs\t\toffset was " << sweepCut.offset  << std::endl;

    std::vector<NodeIx> expectedLabels = getLabels(graph, part);

    t.start();
    cuda.computeSweepCuts();
    cuda.cutClusters();
    auto timeGPU = t.timeMicros();
    std::cout << "GPU time: " << timeGPU << "μs" << std::endl;

    std::cout << "\t -> speedup: " << (static_cast<float>(timeCPU) / timeGPU) << std::endl;


    AllSweepCuts result = cuda.readSweepCuts();
    EXPECT_NEAR(result.cuts[0].sparsity, sweepCut.sparsity, 0.00000001);

    // std::vector<NodeData> pt = cuda.downloadPartition();

    FinalPartition fp = cuda.getFinalPartition();

    for (NodeIx nix = 0; nix < graph.numNodes; nix++) {
        EXPECT_EQ(fp.clusterIds[nix], expectedLabels[nix]) << " nix: " << nix;
    }
}

inline int hashVector(const std::vector<NodeIx>& vec) {
    unsigned int sum = 0;

    for (unsigned int x : vec) {
        unsigned int h = x * 0x45d9f3b;
        h = ((h >> 16) ^ h) * 0x45d9f3b;
        h = ((h >> 16) ^ h);
        sum += h;
    }

    return static_cast<int>(sum & 0x7FFFFFFF);
}

std::unordered_map<int, int> gpuLookup(const std::vector<NodeData>& pt) {

    std::unordered_map<int, std::vector<NodeIx>> clusters;

    for (const NodeData& nd: pt) {
        clusters[nd.label].push_back(nd.nix);
    }

    std::unordered_map<int, int> lookup;

    for (auto& [label, nodes]: clusters) {
        lookup[hashVector(nodes)] = label;
    }

    return lookup;
}


TEST_P(CudaTest, RepeatedCuts) {
    resetGraph();

    auto rwData = cuda.readRandomWalkValues();
    RandomWalk rw(graph.numNodes);
    rw.setData(rwData);
    Partition part(&graph);

    std::vector<NodeIx> active, next{0};

    for(int i = 0; i < GetParam(); i++) {
        std::cout << "=====================================================================================" << std::endl;

        std::swap(active, next);
        next.clear();


        auto deg = cuda.downloadDegrees();
        for(NodeIx nix = 0; nix < graph.numNodes; nix++) {
            ASSERT_EQ(part.vertexFor(nix).nix, nix);
            ASSERT_EQ(part.vertexFor(nix).internalDegree, deg[nix]) << " nix = " << nix << "\t Iteration: " << (i+1);
        }

        // pt = cuda.downloadPartition();
        // for (NodeIx cid = 0; cid < part.numClusters(); cid++) {
        //     for (const ClusterVertex& cv: part.getCluster(cid)) {
        //         ASSERT_EQ(pt[cv.nix].label, cid) << " nix: " << cv.nix << "\t Iteration: " << (i+1);
        //     }
        // }

        rw.iterate(part, active);
        cuda.iterateRandomWalk();

        auto y = cuda.readRandomWalkValues();
        auto z = rw.values();

        // pt = cuda.downloadPartition();
        // for (NodeIx cid = 0; cid < part.numClusters(); cid++) {
        //     for (const ClusterVertex& cv: part.getCluster(cid)) {
        //         ASSERT_EQ(pt[cv.nix].label, cid) << " nix: " << cv.nix << "\t Iteration: " << (i+1);
        //     }
        // }

        for (NodeIx cix: active) {
            const Cluster& cluster = part.getCluster(cix);
            for (const ClusterVertex& cv: cluster) {
                EXPECT_NEAR(y[cv.nix], z[cv.nix], 0.00001) << " nix = " << cv.nix << "\t Iteration: " << (i+1);
            }
        }

        // TODO active should remove inactive clusters, then only check active nodes for random walk equality!!

        cuda.computeSweepCuts();
        auto gpuSweepCuts = cuda.readSweepCuts();

        pt = cuda.downloadPartition();

        std::unordered_map<int, int> lookup = gpuLookup(pt);

        for(NodeIx ix = 0; ix < active.size(); ix++) {
            std::vector<NodeIx> m;
            part.consolidate(ix, active);
        }


        NodeIx numClusters = active.size();
        for(NodeIx ix = 0; ix < numClusters; ix++) {
            NodeIx clusterId =  active[ix];
            SweepCut sweepCut = part.sweepCut(clusterId, z);

            const Cluster& currentCluster = part.getCluster(clusterId);
            std::vector<NodeIx> nodesInCluster;
            nodesInCluster.reserve(currentCluster.size());
            for (const ClusterVertex& cv: currentCluster) {
                nodesInCluster.push_back(cv.nix);
            }
            int cpuHash = hashVector(nodesInCluster);
            // std::cout << "cpuHash: " << cpuHash << std::endl;

            auto it = lookup.find(cpuHash);
            ASSERT_NE(it, lookup.end());
            int gpuLabel = it->second;

            if (gpuLabel == 316) {

                int ix = 0;
                for (NodeIx nix: nodesInCluster) {
                    EdgeIx edgeDiff = sweepCut.pS[ix].edgeDiff;
                    EdgeIx vol = sweepCut.pS[ix].vol;
                    EdgeIx clusterVol = part.getCluster(clusterId).internalVolume;


                    EdgeIx denom = (std::min(vol, clusterVol - vol));
                    frac_t sparsity = static_cast<frac_t>(edgeDiff) / static_cast<frac_t>(denom);

                    // printf("Node %d has label %d at offset ???, rwvalue = %f, prefixSum = %d and prefixVol = %d -> sparsity: %f, denom = %d, clusterVol = %d\n", nix, gpuLabel, z[nix], edgeDiff, vol, sparsity, denom, clusterVol);
                    ix++;
                }
            }

            // check equal sparsity cuts
            for (int j = 0; j < gpuSweepCuts.clusterIds.size(); j++) {
                if (gpuSweepCuts.clusterIds[j] == gpuLabel) {
                    ASSERT_NEAR(gpuSweepCuts.cuts[j].sparsity, sweepCut.sparsity, 0.00000001) << "GPU label: " << gpuLabel;
                    if (sweepCut.sparsity < 1 && gpuSweepCuts.cuts[j].sparsity < 1) {
                        // EXPECT_EQ(gpuSweepCuts.cuts[j].offset - 1, sweepCut.offset);
                        sweepCut.offset = gpuSweepCuts.cuts[j].offset - 1;
                    }
                    break;
                }
            }

            if (sweepCut.sparsity >= sc_threshold) continue;
            part.split<false, true>(clusterId, sweepCut.offset, active);
        }

        for (NodeIx cix = 0; cix < part.numClusters(); cix++) {
            float potential = rw.recenterCluster(part.getCluster(cix));
            if (potential >= rw_threshold) {
                next.push_back(cix);
            }
        }

        std::vector<NodeIx> expectedLabels = getLabels(graph, part);


        cuda.cutClusters();
        pt = cuda.downloadPartition();

        // checks
        for (NodeIx nix = 0; nix < graph.numNodes; nix++) {
            ASSERT_EQ(pt[nix].nix, nix);
            printf("Node %d is in cluster %d\n", nix, pt[nix].label);
            // EXPECT_EQ(pt[nix].label, expectedLabels[nix]) << " nix: " << nix << "\t Iteration: " << (i+1);
        }

        for (NodeIx cix = 0; cix < part.numClusters(); cix++) {
            ClusterVertex first = *part.getCluster(cix).begin();
            NodeIx label = pt[first.nix].label;
            for (const ClusterVertex& cv: part.getCluster(cix)) {
                ASSERT_EQ(pt[cv.nix].label, label) << "nix " << cv.nix << " should be in cluster " << label << ", but is in " << pt[cv.nix].label << ". First is " << first.nix  << "\t Iteration: " << (i+1);
            }
        }


        y = cuda.readRandomWalkValues();
        z = rw.values();
        for (NodeIx cix: next) {
            const Cluster& cluster = part.getCluster(cix);
            for (const ClusterVertex& cv: cluster) {
                EXPECT_NEAR(y[cv.nix], z[cv.nix], 0.00001) << " nix = " << cv.nix << "\t Iteration: " << (i+1);
            }
        }


        // for (NodeIx nix = 0; nix < graph.numNodes; nix++) {
        //     NodeIx ownLabel = pt[nix].label;
        //     for(auto it = graph.nbegin(nix); it != graph.nend(nix); ++it) {
        //         NodeIx nb = *it;
        //         NodeIx nbLabel = pt[nb].label;
        //         int edgeActive = (nbLabel == ownLabel) ? 1 : 0;
        //         EdgeIx eix = std::distance(graph.edges.begin(), it);
        //         if (eix == 2) {
        //             const auto& neighbors = graph.edges;
        //             printf("CPU: %d (%d) <-> %d (%d)\n", nix, ownLabel, nb, nbLabel);
        //             printf("CPU-neighbors: %d %d %d %d %d %d %d\n", neighbors[0], neighbors[1], neighbors[2], neighbors[3], neighbors[4], neighbors[5], neighbors[6]);
        //         }
        //
        //         ASSERT_EQ(aem[eix], edgeActive) << " eix = " << eix << "\n labels: " << ownLabel << " <-> " << nbLabel;
        //     }
        // }
    }
}


TEST_F(CudaTest, ExpanderDecomposition) {
    GTEST_SKIP();
    Timer t;
    t.start();
    cuda.expanderDecomposition();

    printf("Terminated after %fs\n", t.timeSeconds());
}


*/


