//
// Created by robin on 19.01.2026.
//

#ifndef PAREX_GRAPH_MANAGER_CU
#define PAREX_GRAPH_MANAGER_CU

#include "interface.h"
#include "devRandomWalk.h"

#include <thrust/device_vector.h>

template<typename T>
inline void copyToDevice(const std::vector<T>& elements, T* devTarget, cudaStream_t stream) {
    cudaMemcpyAsync(
            devTarget, elements.data(), elements.size() * sizeof(T), cudaMemcpyHostToDevice, stream
    );
}

class CudaDeviceManager::Impl {

    std::unique_ptr<GraphManager> gm;
    std::unique_ptr<RandomWalkManager> rw;

public:

    void initialize(const Graph& graph) {
        // 1. Clear existing if re-initializing
        gm.reset();
        rw.reset();

        gm = std::make_unique<GraphManager>(graph);
        rw = std::make_unique<RandomWalkManager>(gm->getView(), graph.numNodes);
    }

//    void initialize(const Graph& graph) {
//
//        dist.resize(n);
//        old_dist.resize(n);
//        node_val.resize(n);
//        initRandomWalk(dist, n);
//        segSum.init(graphView(), randomWalkView(), n);
//    }
//
    void iterateRandomWalk() {
        rw->step(gm->getView());
    }

    std::vector<frac_t> readRandomWalkValues() {
        return rw->readRandomWalkValues();
    }
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

void CudaDeviceManager::iterateRandomWalk() { impl->iterateRandomWalk(); }

//void CudaDeviceManager::applyGraphUpdates(const std::vector<EdgeIx>& edgeDeletions, const std::vector<NodeUpdate>& updates) {
//    impl->applyGraphUpdates(edgeDeletions, updates);
//}


#endif //PAREX_GRAPH_MANAGER_CU
