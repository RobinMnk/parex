//
// Created by robin on 19.01.2026.
//

#ifndef PAREX_GRAPH_MANAGER_CU
#define PAREX_GRAPH_MANAGER_CU

#include "interface.h"
#include "devRandomWalk.h"
#include "devSweepCut.h"
#include "devPartition.h"
#include "timer.h"

struct CudaDeviceManager::Impl {
    std::unique_ptr<GraphManager> gm;
    std::unique_ptr<PartitionManager> pt;
    std::unique_ptr<RandomWalkManager> rw;
    std::unique_ptr<SweepCutManager> sc;

    void initialize(const Graph& graph) {
        gm.reset();
        pt.reset();
        rw.reset();
        sc.reset();

        gm = std::make_unique<GraphManager>(graph);
        sc = std::make_unique<SweepCutManager>(graph.numNodes);
        pt = std::make_unique<PartitionManager>(*gm, sc->getKeyBuffer());
        rw = std::make_unique<RandomWalkManager>(graph.numNodes);
    }

    void iterateRandomWalk() {
        rw->stepFast(*gm, pt->getPartitionView(), sc->getKeyBuffer(), pt->getActiveDegrees());
    }

    std::vector<frac_t> readRandomWalkValues() {
        return rw->valuesToCPU();
    }

    std::vector<EdgeIx> downloadDegrees() {
        return pt->downloadActiveDegrees();
    }

    void computeSweepCuts() {
        sc->compute(*gm, *pt, rw->randomWalkValues());
    }

    void cutClusters() {
        pt->cutClusters(sc->getSweepCuts());

        pt->computeClusterData();

        pt->recenterAndDeactivateClusters(rw->getValues());

        // absolutely crucial!!
        fixupPartition();

        pt->disableEdges(*gm);

        pt->computeActiveDegrees(*gm);
    }

    inline void expanderDecomposition();

    void fixupPartition() {
        pt->scatter();
    }

    AllSweepCuts getSweepCuts() {
        NodeIx numActiveClusters = pt->numActiveClusters;
        std::vector<int> clusterIds(numActiveClusters);
        std::vector<SweepCutData> cuts(numActiveClusters);
        thrust::copy(pt->getActiveLabels().begin(), pt->getActiveLabels().begin() + numActiveClusters, clusterIds.begin());
        thrust::copy(sc->getSweepCuts().begin(), sc->getSweepCuts().begin() + numActiveClusters, cuts.begin());
        return {clusterIds, cuts};
    }


    std::vector<NodeData> downloadPartition() {
        return pt->downloadPartition();
    }

    std::vector<int> downloadActiveEdgeMap() {
        return pt->getActiveEdgeMap(*gm);
    }

    void updateLabels(std::vector<NodeIx>& nodeLabels, NodeIx activeClusters) {
        gm->updateLabels(nodeLabels, activeClusters);
    }

    Graph downloadGraph() {
        return gm->downloadGraph();
    }

    void printInf(
        std::vector<int>& labels,
        std::vector<ClusterData>& clusters,
        std::vector<EdgeIx>& degs,
        std::vector<SweepCutData>& scs,
        std::vector<NodeData>& nodes
    );
};


void printNodes(std::vector<NodeData>& nodes, NodeIx n, NodeData* ptr) {
    thrust::device_ptr<NodeData> dev_ptr2(ptr);
    thrust::copy(dev_ptr2, dev_ptr2 + n, nodes.begin());

    for (auto j = 0; j < n; j++) {
        std::cout << "(" << nodes[j].nix << ", " << nodes[j].label  << "), ";
    }
    std::cout << std::endl;
}



// void CudaDeviceManager::Impl::expanderDecomposition() {
//     std::vector<int> labels(gm->n);
//     // std::vector<ClusterData> clusters(gm->n);
//     // std::vector<EdgeIx> degs(gm->n);
//     // std::vector<SweepCutData> scs(gm->n);
//     std::vector<NodeData> nodes(gm->n);
//
//     int numClusters = sc->getNumActiveClusters();
//
//     while (labels[numClusters-1] >= 0) {
//
//         // printf("====================================================================================\n");
//
//         auto& x = sc->getLabels();
//         thrust::copy(x.begin(), x.end(), labels.begin());
//         for (int y = 0; y < sc->getNumActiveClusters(); y++) std::cout << labels[y] << ", ";
//         std::cout << std::endl;
//
//
//         iterateRandomWalk();
//         computeSweepCuts();
//
//         // printf("Cutting clusters\n");
//         pt->cutClusters(sc->getSweepCuts(), sc->getLabels(), sc->getNumActiveClusters());
//
//         // printf("Recenter and Deactivate\n");
//         rw->recenterAndDeactivateClusters(pt->getPartitionView());
//
//         // absolutely crucial!!
//         fixupPartition();
//
//         pt->computeActiveDegrees(*gm);
//
//         numClusters = sc->getNumActiveClusters();
//     }
//
//     printf("====================================================================================\n");
//
//     printNodes(nodes, gm->n, pt->getPartitionView().Current());
// }


// void CudaDeviceManager::Impl::printInf(
//     std::vector<int>& labels,
//     std::vector<ClusterData>& clusters,
//     std::vector<EdgeIx>& degs,
//     std::vector<SweepCutData>& scs,
//     std::vector<NodeData>& nodes
// ) {
//     auto& x = sc->getLabels();
//     thrust::copy(x.begin(), x.end(), labels.begin());
//     for (int y = 0; y < sc->getNumActiveClusters(); y++) std::cout << labels[y] << ", ";
//     std::cout << std::endl;
//
//     // rw->computeClusterData(pt->getPartitionView(), sc->getLabels());
//     auto& y = rw->getClusterData();
//     thrust::copy(y.begin(), y.begin() + sc->getNumActiveClusters(), clusters.begin());
//
//     auto& u = sc->getSweepCuts();
//     thrust::copy(u.begin(), u.begin() + sc->getNumActiveClusters(), scs.begin());
//
//     auto& z = pt->getActiveDegrees();
//     thrust::copy(z.begin(), z.end(), degs.begin());
//
//     thrust::device_ptr<NodeData> dev_ptr(pt->getPartitionView().Current());
//     thrust::copy(dev_ptr, dev_ptr + gm->n, nodes.begin());
//
//     for (auto i = 0; i < sc->getNumActiveClusters(); i++) {
//         std::cout << "cluster " << labels[i] << ":\n"
//                      "\tpotential: " << clusters[i].maxPotential - clusters[i].minPotential <<
//                          "\n\taverage:   " << clusters[i].rwSum / clusters[i].totalElements <<
//                          "\n\telements:  " << clusters[i].totalElements <<
//                             "\n\tsc-id:     " << scs[i].clusterId <<
//                          "\n\tsc-spars:  " << scs[i].sparsity <<
//                          "\n\tsc-offs:  " << scs[i].offset
//         << std::endl;
//     }
//     std::cout << "\n" << std::endl;
// }


void CudaDeviceManager::Impl::expanderDecomposition() {
    // std::vector<int> labels(gm->n);
    // std::vector<ClusterData> clusters(gm->n);
    // std::vector<EdgeIx> degs(gm->n);
    // std::vector<SweepCutData> scs(gm->n);
    // std::vector<NodeData> nodes(gm->n);

    int i = 0;

    Timer t;
    t.start();
    while (i++ < 2000 && pt->numActiveClusters > 0) {
        // printf("==================================================================================== It: %d\n", (i+1));

        iterateRandomWalk();
        computeSweepCuts();
        cutClusters();
    }

    auto tm = t.timeSeconds();
    printf("%d iterations with %fs per iteration\n", i, (float) tm / i);


    // printf("Before cutting\n");
    // printInf(labels, clusters, degs, scs, nodes);

        // printf("After cutting\n");
        // printInf(labels, clusters, degs, scs, nodes);



        // auto& x = sc->getLabels();
        // thrust::copy(x.begin(), x.end(), labels.begin());
        // for (int y = 0; y < sc->getNumActiveClusters(); y++) std::cout << labels[y] << ", ";
        // std::cout << std::endl;

        // for (auto j = 0; j < gm->n; j++) {
        //     std::cout << "(" << nodes[j].nix << ", " << nodes[j].label  << ", " << nodes[j].activeDegree << "), ";
        // }
        //

        // printf("Cutting clusters\n");
        // pt->cutClusters(sc->getSweepCuts(), sc->getLabels(), sc->getNumActiveClusters());
        //
        //
        // printf("Recenter and Deactivate\n");
        // rw->recenterAndDeactivateClusters(pt->getPartitionView());
        //
        // // absolutely crucial!!
        // fixupPartition();
        //
        // pt->computeActiveDegrees(*gm);

    // }
}




CudaDeviceManager::CudaDeviceManager() : impl(std::make_unique<Impl>()) {}

CudaDeviceManager::~CudaDeviceManager() = default;

void CudaDeviceManager::initialize(const Graph &graph) { impl->initialize(graph); }

Graph CudaDeviceManager::downloadGraph() { return impl->downloadGraph(); }

std::vector<frac_t> CudaDeviceManager::readRandomWalkValues() { return impl->readRandomWalkValues(); }
std::vector<EdgeIx> CudaDeviceManager::downloadDegrees() { return impl->downloadDegrees(); }

void CudaDeviceManager::iterateRandomWalk() { impl->iterateRandomWalk(); }

void CudaDeviceManager::computeSweepCuts() { impl->computeSweepCuts(); }

void CudaDeviceManager::cutClusters() { impl->cutClusters(); }

void CudaDeviceManager::fixupPartition() { impl->fixupPartition(); }

void CudaDeviceManager::expanderDecomposition() { impl->expanderDecomposition(); }

AllSweepCuts CudaDeviceManager::readSweepCuts() { return impl->getSweepCuts(); }

std::vector<NodeData> CudaDeviceManager::downloadPartition() { return impl->downloadPartition(); }

std::vector<int> CudaDeviceManager::downloadActiveEdgeMap() { return impl->downloadActiveEdgeMap(); }

void CudaDeviceManager::updateLabels(std::vector<NodeIx>& nodeLabels, NodeIx activeClusters) { impl->updateLabels(nodeLabels, activeClusters); }


//void CudaDeviceManager::inspectSweepCut(std::vector<EdgeIx>& prefixSums, std::vector<EdgeIx>& cutVolumes) { impl->inspectSweepCut(prefixSums, cutVolumes); }

//void CudaDeviceManager::applyGraphUpdates(const std::vector<EdgeIx>& edgeDeletions, const std::vector<NodeUpdate>& updates) {
//    impl->applyGraphUpdates(edgeDeletions, updates);
//}


#endif //PAREX_GRAPH_MANAGER_CU
