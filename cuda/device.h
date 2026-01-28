//
// Created by robin on 19.01.2026.
//

#ifndef PAREX_GRAPH_MANAGER_CU
#define PAREX_GRAPH_MANAGER_CU

#include "interface.h"
#include "devRandomWalk.h"
#include "devSweepCut.h"
#include "devPartition.h"

template<typename T>
inline void copyToDevice(const std::vector<T>& elements, T* devTarget, cudaStream_t stream) {
    cudaMemcpyAsync(
            devTarget, elements.data(), elements.size() * sizeof(T), cudaMemcpyHostToDevice, stream
    );
}

class CudaDeviceManager::Impl {
    std::unique_ptr<GraphManager> gm;
    std::unique_ptr<PartitionManager> pt;
    std::unique_ptr<RandomWalkManager> rw;
    std::unique_ptr<SweepCutManager> sc;

public:

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
        rw->step(*gm, thrust::raw_pointer_cast(pt->partition1.data()), thrust::raw_pointer_cast(sc->packedKeysIn.data()));
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

    AllSweepCuts getSweepCuts() {
        return sc->resultToCPU(gm->numClusters);
    }

    std::vector<NodeData> downloadPartition() {
        return pt->downloadPartition();
    }

    void updateLabels(std::vector<NodeIx>& nodeLabels, NodeIx activeClusters) {
        gm->updateLabels(nodeLabels, activeClusters);
    }

//    void inspectSweepCut(std::vector<EdgeIx>& prefixSums, std::vector<EdgeIx>& cutVolumes) {
//        sc->inspect(prefixSums, cutVolumes);
//    }


//
//    void applyGraphUpdates(const std::vector<EdgeIx>& edgeDeletions, const std::vector<NodeUpdate>& updates) {
//        DevGraph device = graphView();
//        if (!edgeDeletions.empty()) {
//            copyToDevice(edgeDeletions, device.edgeDeletionBuffer, nullptr);
//            auto elements = static_cast<NodeIx>(edgeDeletions.size());
//            size_t blocks = (elements + threads - 1) / threads;
//            deactivateEdgeKernel<<<blocks, threads, 0, nullptr>>>(device, elements);
//
//            // Check if the launch itself was valid
//            cudaError_t err = cudaGetLastError();
//            if (err != cudaSuccess) {
//                printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
//            }
//        }
//
//        if (!updates.empty()) {
//            copyToDevice(updates, device.nodeUpdateBuffer, nullptr);
//            auto elements = static_cast<NodeIx>(updates.size());
//            size_t blocks = (elements + threads - 1) / threads;
//            degreeKernel<<<blocks, threads, 0, nullptr>>>(device, elements);
//        }
//
//        cudaError_t err = cudaStreamSynchronize(nullptr);
//        if (err != cudaSuccess) {
//            fprintf(stderr, "GPU Error: %s\n", cudaGetErrorString(err));
//        }
//    }

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

AllSweepCuts CudaDeviceManager::readSweepCuts() { return impl->getSweepCuts(); }

std::vector<NodeData> CudaDeviceManager::downloadPartition() { return impl->downloadPartition(); }

void CudaDeviceManager::updateLabels(std::vector<NodeIx>& nodeLabels, NodeIx activeClusters) { impl->updateLabels(nodeLabels, activeClusters); }


//void CudaDeviceManager::inspectSweepCut(std::vector<EdgeIx>& prefixSums, std::vector<EdgeIx>& cutVolumes) { impl->inspectSweepCut(prefixSums, cutVolumes); }

//void CudaDeviceManager::applyGraphUpdates(const std::vector<EdgeIx>& edgeDeletions, const std::vector<NodeUpdate>& updates) {
//    impl->applyGraphUpdates(edgeDeletions, updates);
//}


#endif //PAREX_GRAPH_MANAGER_CU
