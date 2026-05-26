//
// Created by robin on 19.01.2026.
//

#ifndef PAREX_GRAPH_MANAGER_CU
#define PAREX_GRAPH_MANAGER_CU

#include "devConsolidate.h"
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
    std::unique_ptr<ConsolidationManager> cm;

    void initialize(const Graph& graph) {
        gm.reset();
        pt.reset();
        rw.reset();
        sc.reset();
        cm.reset();

        gm = std::make_unique<GraphManager>(graph);
        sc = std::make_unique<SweepCutManager>(graph.numNodes);
        pt = std::make_unique<PartitionManager>(*gm);
        rw = std::make_unique<RandomWalkManager>(*gm);
        cm = std::make_unique<ConsolidationManager>(graph.numNodes, 2 * graph.numEdges);
    }

    void iterateRandomWalk() const {
        rw->stepFast(
            *gm,
            pt->getActiveNodes(),
            sc->getKeyBuffer(),
            pt->getAllInternalDegrees(),
            pt->numActiveNodes
        );
    }

    std::vector<frac_t> readRandomWalkValues() const {
        return rw->valuesToCPU();
    }

    std::vector<EdgeIx> downloadDegrees() const {
        return pt->downloadActiveDegrees();
    }

    int getNumClusters() {
        return 0;
        // return pt->numActiveClusters;
    }

    EdgeIx getNumCutEdges() {
        return pt->numCutEdges(*gm);
    }

    FinalPartition getFinalPartition() {
        return pt->finalizePartition();
    }

    std::vector<int> downloadLabels() {
        std::vector<int> labels(gm->n);
        thrust::copy(pt->getAllLabels().begin(), pt->getAllLabels().end(), labels.begin());
        return labels;
    }

    void computeSweepCuts() const {
        sc->compute(
            *gm, *pt, rw->getValues()
        );
    }

    void cutClusters() {
        // TODO: if there are no valid sweep cuts (all above the threshold) -> continue!
        if (sc->numClustersWithCut == 0) return;

        inspect(pt->getUniqueActiveLabels(), pt->numActiveClusters);

        pt->cutClusters(sc->getScNodeData(), sc->getSweepCuts(), sc->numClustersWithCut);

        inspect(pt->getUniqueActiveLabels(), pt->numActiveClusters);

        // std::vector<NodeData> nodes(gm->n);
        // thrust::device_ptr<NodeData> dev_ptr(pt->getPartitionView().Current());
        // thrust::copy(dev_ptr, dev_ptr + gm->n, nodes.begin());
        // printf("Before Consolidate\n");
        // for (NodeIx i = 0; i < gm->n; i++) {
        //     printf("Node %d has label %d\n", nodes[i].nix, nodes[i].label);
        // }

        // cm->consolidate(*gm, pt->getActiveNodes(), pt->numActiveNodes);

        // thrust::copy(dev_ptr, dev_ptr + gm->n, nodes.begin());
        // printf("After Consolidate\n");
        // for (NodeIx i = 0; i < gm->n; i++) {
        //     printf("Node %d has label %d\n", nodes[i].nix, nodes[i].label);
        // }


        pt->computeClusterData(rw->getValues());


        inspect(pt->getUniqueActiveLabels(), pt->numActiveClusters);

        pt->recenterAndDeactivateClusters(rw->getValues());

        inspect(pt->getUniqueActiveLabels(), pt->numActiveClusters);

        // absolutely crucial!!
        // fixupPartition();

        // thrust::copy(dev_ptr, dev_ptr + gm->n, nodes.begin());
        // printf("After Fixup\n");
        // for (NodeIx i = 0; i < gm->n; i++) {
        //     printf("Node %d has label %d\n", nodes[i].nix, nodes[i].label);
        // }
        // std::cout << std::endl;

        pt->disableEdges(*gm);

        pt->computeActiveDegrees(*gm);
    }

    inline void expanderDecomposition();

    AllSweepCuts getSweepCuts() {
        NodeIx numActiveClusters = pt->numActiveClusters;
        std::vector<int> clusterIds(numActiveClusters);
        std::vector<SweepCutData> cuts(numActiveClusters);
        thrust::copy(pt->getUniqueActiveLabels().begin(), pt->getUniqueActiveLabels().begin() + numActiveClusters, clusterIds.begin());
        thrust::copy(sc->getSweepCuts().begin(), sc->getSweepCuts().begin() + numActiveClusters, cuts.begin());
        return {clusterIds, cuts};
    }


    std::vector<NodeData> downloadPartition() {
        return sc->downloadPartition();
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
        std::cout << "(" << nodes[j].nix  << "), ";
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
    const int MAX_NUM_ITER = 200000;
    // const int NUM_RW_STEPS = 4;

    // int walkReset = 128;

    Timer t;
    t.start();
    while (i++ < MAX_NUM_ITER && pt->numActiveClusters > 0 && pt->numActiveNodes > 0) {
        std::cout << "==================================================================================== \n Begin Iteration: " <<  (i) << std::endl;

        // #pragma unroll
        // for (int x = 0; x < NUM_RW_STEPS; x++) {

        // if (i % walkReset == 0) {
        //     rw->initRandomWalk(i);
        //     for (int x = 0; x < walkReset / 4; i++) {
        //         iterateRandomWalk();
        //     }
        //     walkReset *= 2;
        // }

        std::cout << ("Random Walk Step") << std::endl;
        iterateRandomWalk();
        // }

        std::cout << ("Compute SweepCuts") << std::endl;
        computeSweepCuts();

        std::cout << ("Adjust Partition") << std::endl;
        cutClusters();
    }
    if (i >= MAX_NUM_ITER) {
        printf("Not terminating. ABORT.\n");
    }


    auto tm = t.timeSeconds();
    // printf("%d iterations with %fs per iteration\n", i, (float) tm / i);
    printf("iterations: %d\n", i);


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

void CudaDeviceManager::expanderDecomposition() { impl->expanderDecomposition(); }

AllSweepCuts CudaDeviceManager::readSweepCuts() { return impl->getSweepCuts(); }

std::vector<NodeData> CudaDeviceManager::downloadPartition() { return impl->downloadPartition(); }

std::vector<int> CudaDeviceManager::downloadActiveEdgeMap() { return impl->downloadActiveEdgeMap(); }

std::vector<int> CudaDeviceManager::downloadLabels() { return impl->downloadLabels(); }
int CudaDeviceManager::getNumClusters() { return impl->getNumClusters(); }


FinalPartition CudaDeviceManager::getFinalPartition() { return impl->getFinalPartition(); }

EdgeIx CudaDeviceManager::getNumCutEdges() { return impl->getNumCutEdges(); }

void CudaDeviceManager::updateLabels(std::vector<NodeIx>& nodeLabels, NodeIx activeClusters) { impl->updateLabels(nodeLabels, activeClusters); }


//void CudaDeviceManager::inspectSweepCut(std::vector<EdgeIx>& prefixSums, std::vector<EdgeIx>& cutVolumes) { impl->inspectSweepCut(prefixSums, cutVolumes); }

//void CudaDeviceManager::applyGraphUpdates(const std::vector<EdgeIx>& edgeDeletions, const std::vector<NodeUpdate>& updates) {
//    impl->applyGraphUpdates(edgeDeletions, updates);
//}


#endif //PAREX_GRAPH_MANAGER_CU
