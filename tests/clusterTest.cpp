//
// Created by robin on 13.03.2025.
//

#include "../lib/utils/io.h"
#include "../lib/core/partition.h"
#include <gtest/gtest.h>
#include <numeric>
#include <random>

void checkCluster2(const Cluster& cluster, const Graph& graph, const Partition& part) {
    EdgeIx sumDegrees = 0, summedVol = 0;
    for(const ClusterVertex cv: cluster) {
        // range begins correctly, and end is within correct range
        ASSERT_EQ(cluster.edgeBegin(cv), graph.nbegin(cv.nix));
        ASSERT_EQ(cv.end, std::distance(graph.edges.begin(), graph.nbegin(cv.nix) + cv.internalDegree));
        ASSERT_LE(cv.end, std::distance(graph.edges.begin(), graph.nend(cv.nix)));
        ASSERT_EQ(cv.nix, part.vertexFor(cv.nix).nix);

        // check outgoing edges
        for(auto it = cluster.edgeBegin(cv); it != cluster.edgeEnd(cv); ++it) {
            // edge exists in graph
            auto itx = std::find_if(graph.nbegin(cv.nix), graph.nend(cv.nix),
                                    [&it](NodeIx otherNix) { return otherNix == *it; } );
            ASSERT_NE(itx, graph.nend(cv.nix));

            // outgoing edge ends in the same cluster (i.e., edge is actually active)
            auto it2 = std::find_if(cluster.begin(), cluster.end(),
                                    [&it](const ClusterVertex& node) { return node.nix == *it; } );
            ASSERT_NE(it2, cluster.end());
        }

        // internalDegree matches range
        ASSERT_EQ(cv.internalDegree, std::distance(cluster.edgeBegin(cv), cluster.edgeEnd(cv)));
        sumDegrees += cv.internalDegree;
        summedVol += graph.degree(cv.nix);
    }
    ASSERT_EQ(summedVol, cluster.volume);
    ASSERT_EQ(sumDegrees, cluster.internalVolume);
}

typedef size_t TestParam;

class ClusterTest : public ::testing::TestWithParam<TestParam> {
protected:
    static Graph graph;
    static std::mt19937 gen;
    static std::random_device rd;

    template<typename T>
    static void randomPermutation(std::vector<T>& v) {
        std::mt19937 g(rd());
        std::shuffle(v.begin(), v.end(), g);
    }

    [[maybe_unused]] static void SetUpTestSuite() {
//        graph = readGraph();
        graph = readDynGraph("../graphs/uk.mtx").finalize();
    }

    static void splitAndCheck(Partition& partition, std::vector<NodeIx> values, NodeIx clusterId, size_t k) {
        const Cluster& cluster = partition.getCluster(clusterId);
        EdgeIx origVol = cluster.volume, origIntVol = cluster.internalVolume;

        size_t ix = 0;
        for(ClusterVertex cv: cluster) {
            values[cv.nix] = (k * ix) / cluster.size();
            ix++;
        }
        randomPermutation(values);

        std::vector<NodeIx> modified{clusterId};
        EdgeIx cutEdges = partition.splitByIndices<true, false>(clusterId, values, k, modified);

        ASSERT_EQ(modified.size(), k);

        EdgeIx summedVol{0}, summedInternalVol{0};
        for(size_t i: modified) {
            const Cluster& cl = partition.getCluster(i);
            checkCluster2(cl, graph, partition);
            summedVol += cl.volume;
            summedInternalVol += cl.internalVolume;
        }

        ASSERT_EQ(summedVol, origVol);
        ASSERT_EQ(summedInternalVol + 2 * cutEdges, origIntVol);
    }
};
Graph ClusterTest::graph{};
std::random_device ClusterTest::rd{};
std::mt19937 ClusterTest::gen{23121};

TEST_F(ClusterTest, CreateFullPartition) {
    Partition partition{graph};
    ASSERT_EQ(partition.numClusters(), 1);
    ASSERT_EQ(partition.getCluster(0).volume, 2 * graph.numEdges);
    ASSERT_EQ(partition.getCluster(0).internalVolume, 2 * graph.numEdges);
}

TEST_F(ClusterTest, CheckFullPartition) {
    Partition partition{graph};
    Cluster cluster = partition.getCluster(0);
    EdgeIx sumDegrees = 0, summedVol = 0;
    for(ClusterVertex cn: cluster) {
        ASSERT_EQ(cluster.edgeBegin(cn), graph.nbegin(cn.nix));
        ASSERT_EQ(cluster.edgeEnd(cn), graph.nend(cn.nix));
        ASSERT_EQ(cn.internalDegree, cn.end - std::distance(graph.edges.begin(), cluster.edgeBegin(cn)));
        sumDegrees += cn.internalDegree;
        summedVol += graph.degree(cn.nix);
    }
    ASSERT_EQ(summedVol, cluster.volume);
    ASSERT_EQ(sumDegrees, cluster.internalVolume);
}

void checkCluster(Graph& graph, Cluster& cluster, Cluster& other) {
    EdgeIx sumDegrees = 0, summedVol = 0;
    for(ClusterVertex cn: cluster) {
        // this node is not also in the other cluster
        auto itx = std::find_if(other.begin(), other.end(),
                                [&cn](const ClusterVertex& node) { return node.nix == cn.nix; } );
        ASSERT_EQ(itx, other.end());

        // range begins correctly, and end is within correct range
        ASSERT_EQ(cluster.edgeBegin(cn), graph.nbegin(cn.nix));
        ASSERT_LE(cn.end, std::distance(graph.edges.begin(), graph.nend(cn.nix)));

        // check outgoing edges
        for(auto it = cluster.edgeBegin(cn); it != cluster.edgeEnd(cn); ++it) {
            // edge exists in graph
            auto itG = std::find_if(graph.nbegin(cn.nix), graph.nend(cn.nix),
                                    [&it](NodeIx otherNix) { return otherNix == *it; } );
            ASSERT_NE(itG, graph.nend(cn.nix));

            // outgoing edge ends in the same cluster
            itx = std::find_if(cluster.begin(), cluster.end(),
                                    [&itG](const ClusterVertex& node) { return node.nix == *itG; } );
            ASSERT_NE(itx, cluster.end());
        }

        // internalDegree matches range
        ASSERT_EQ(cn.internalDegree, std::distance(cluster.edgeBegin(cn), cluster.edgeEnd(cn)));
        sumDegrees += cn.internalDegree;
        summedVol += graph.degree(cn.nix);
    }
    ASSERT_EQ(summedVol, cluster.volume);
    ASSERT_EQ(sumDegrees, cluster.internalVolume);
}

TEST_F(ClusterTest, TopLevelCuts) {
    for(NodeIx cutThreshold = 1; cutThreshold < graph.numNodes-1; cutThreshold += graph.numNodes / 4) {
        Partition partition{graph};
        std::vector<NodeIx> values(graph.numNodes);
        std::iota(values.begin(), values.end(), 0);
        std::vector<NodeIx> modified{0};

        partition.split<true, false>(0, cutThreshold, modified);

        if(modified.empty()) continue; // no cut was performed

        Cluster left = partition.getCluster(modified[0]);
        Cluster right = partition.getCluster(modified[1]);

        ASSERT_EQ(left.volume + right.volume, 2 * graph.numEdges);
        ASSERT_EQ(left.internalVolume + right.internalVolume + 2 * partition.getNumCutEdges(), 2 * graph.numEdges);    // all edges accounted for
        ASSERT_EQ(left.size() + right.size(), graph.numNodes);      // all nodes accounted for

        checkCluster(graph, left, right);
        checkCluster(graph, right, left);
    }
}

TEST_P(ClusterTest, MultiplePartitions) {
    Partition partition{graph};
    std::vector<NodeIx> values(graph.numNodes);
    size_t k = GetParam();
    splitAndCheck(partition, values, 0, k);
}

TEST_F(ClusterTest, NestedPartitions) {
    Partition partition{graph};
    std::vector<NodeIx> values(graph.numNodes);
    size_t k = 2; // GetParam();
    splitAndCheck(partition, values, 0, k);
    splitAndCheck(partition, values, k/2, 2);
    splitAndCheck(partition, values, k/2, 4);
    splitAndCheck(partition, values, partition.numClusters()-1, 4);
    splitAndCheck(partition, values, partition.numClusters()-1, 2);
}


INSTANTIATE_TEST_SUITE_P(
    P_,
    ClusterTest,
    testing::Values(2, 4, 8, 16, 64, 256)
);