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

class CudaTest : public ::testing::TestWithParam<int> {
protected:
    static Graph graph;
    static CudaDeviceManager cuda;

    [[maybe_unused]] static void SetUpTestSuite() {
        graph = readDynGraph("../../graphs/uk.mtx").finalize();
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

TEST_P(CudaTest, RandomWalk) {
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
//    testing::Values(2)
    testing::Values(0, 1, 2, 4, 8) // , 16, 64, 128, 256)
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
    cuda.fixupPartition();
    AllSweepCuts result = cuda.readSweepCuts();

    std::vector<NodeData> pt = cuda.downloadPartition();

    NodeIx n = graph.numNodes;

//    for(int j = 0; j < 20; j++) {
//        std::cout << "CPU:  " << expected.pS[j].edgeDiff << " / " << expected.pS[j].vol << " (" << (static_cast<float>(expected.pS[j].edgeDiff) / expected.pS[j].vol) << ")" << "\t\t"
//                << "GPU:  " << pt[j].prefixEdgeDiff << " / " << pt[j].prefixVolume << " (" << (static_cast<float>(pt[j].prefixEdgeDiff) / pt[j].prefixVolume) << ")" << std::endl;
//    }

    for(int j = 0; j < n; j++) {
//        EXPECT_EQ(result.prefixSums[j].volume, expected.pS[j].volume);
        ASSERT_EQ(pt[j].label, 0);
        ASSERT_EQ(pt[j].nix, j);
    }

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
        cuda.fixupPartition();
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
    // TODO: this only cuts cluster 0
    SweepCut sweepCut = part.sweepCut(0, z);
    std::vector<NodeIx> modified{0};
    part.split<false, false>(0, sweepCut.offset, modified);
    auto timeCPU = t.timeMicros();
    std::cout << "CPU time: " << timeCPU << "μs" << std::endl;

    std::vector<NodeIx> expectedLabels = getLabels(graph, part);

    t.start();
    cuda.computeSweepCuts();
    cuda.cutClusters();
    auto timeGPU = t.timeMicros();
    std::cout << "GPU time: " << timeGPU << "μs" << std::endl;

    std::cout << "\t -> speedup: " << (static_cast<float>(timeCPU) / timeGPU) << std::endl;


    AllSweepCuts result = cuda.readSweepCuts();
    EXPECT_NEAR(result.cuts[0].sparsity, sweepCut.sparsity, 0.00000001);


    std::vector<NodeData> pt = cuda.downloadPartition();

    for (NodeIx nix = 0; nix < graph.numNodes; nix++) {
        EXPECT_EQ(pt[nix].label, expectedLabels[nix]) << " nix: " << nix;
    }
}


TEST_P(CudaTest, RepeatedCuts) {
    auto rwData = cuda.readRandomWalkValues();
    RandomWalk rw(graph.numNodes);
    rw.setData(rwData);
    Partition part(&graph);

    std::vector<NodeIx> active{0};

    for(int i = 0; i < GetParam(); i++) {
        auto deg = cuda.downloadDegrees();
        for(NodeIx nix = 0; nix < graph.numNodes; nix++) {
            ASSERT_EQ(part.vertexFor(nix).internalDegree, deg[nix]) << " nix = " << nix;
        }

        rw.iterate(part, active);
        cuda.iterateRandomWalk();

        auto y = cuda.readRandomWalkValues();
        auto z = rw.values();

        for (NodeIx nix = 0; nix < graph.numNodes; nix++) {
            ASSERT_NEAR(y[nix], z[nix], 0.00001);
        }

        NodeIx numClusters = active.size();
        for(NodeIx clusterId = 0; clusterId < numClusters; clusterId++) {
            SweepCut sweepCut = part.sweepCut(clusterId, z);
            part.split<false, false>(clusterId, sweepCut.offset, active);
        }

        std::vector<NodeIx> expectedLabels = getLabels(graph, part);

        cuda.computeSweepCuts();
        cuda.cutClusters();

        std::vector<NodeData> pt = cuda.downloadPartition();

        std::cout << "CPU Partition:" << std::endl;
        for(NodeIx clusterId = 0; clusterId < part.numClusters(); clusterId++) {
            auto& cl = part.getCluster(clusterId);
            std::cout << "\t" << clusterId << ": " << cl.size() << " [" << cl.volume << "]" << std::endl;
        }

        for (NodeIx nix = 0; nix < graph.numNodes; nix++) {
            EXPECT_EQ(pt[nix].label, expectedLabels[nix]) << " nix: " << nix;
        }
    }
}



