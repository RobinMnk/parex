//
// Created by robin on 19.01.2026.
//

#ifndef PAREX_GRAPH_MANAGER_CU
#define PAREX_GRAPH_MANAGER_CU

#include "interface.h"
#include "devRandomWalk.h"
#include "devSweepCut.h"
#include "devPartition.h"

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
        pt = std::make_unique<PartitionManager>(*gm);
        rw = std::make_unique<RandomWalkManager>(graph.numNodes);
        sc = std::make_unique<SweepCutManager>(graph.numNodes, *pt);
    }

    void iterateRandomWalk() {
        rw->step(*gm, pt->getPartitionView(), sc->getKeyBuffer(), pt->getActiveDegrees());
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
        pt->cutClusters(sc->getSweepCuts(), sc->getNumActiveClusters());

        // absolutely crucial!!
        fixupPartition();

        pt->computeActiveDegrees(*gm);
    }

    void fixupPartition() {
        pt->scatter();
    }

    AllSweepCuts getSweepCuts() {
        return sc->resultToCPU(gm->numClusters);
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
};

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

AllSweepCuts CudaDeviceManager::readSweepCuts() { return impl->getSweepCuts(); }

std::vector<NodeData> CudaDeviceManager::downloadPartition() { return impl->downloadPartition(); }

std::vector<int> CudaDeviceManager::downloadActiveEdgeMap() { return impl->downloadActiveEdgeMap(); }

void CudaDeviceManager::updateLabels(std::vector<NodeIx>& nodeLabels, NodeIx activeClusters) { impl->updateLabels(nodeLabels, activeClusters); }


//void CudaDeviceManager::inspectSweepCut(std::vector<EdgeIx>& prefixSums, std::vector<EdgeIx>& cutVolumes) { impl->inspectSweepCut(prefixSums, cutVolumes); }

//void CudaDeviceManager::applyGraphUpdates(const std::vector<EdgeIx>& edgeDeletions, const std::vector<NodeUpdate>& updates) {
//    impl->applyGraphUpdates(edgeDeletions, updates);
//}


#endif //PAREX_GRAPH_MANAGER_CU
