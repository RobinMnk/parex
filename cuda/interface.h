//
// Created by robin on 16.01.2026.
//

#ifndef PAREX_INTERFACE_H
#define PAREX_INTERFACE_H

#include "core/graph.h"

struct SwapPair {
    EdgeIx i, j;
};

struct NodeUpdate {
    NodeIx nix;
    EdgeIx diff;
};

class CudaDeviceManager {
public:
    CudaDeviceManager();
    ~CudaDeviceManager();

    CudaDeviceManager(const CudaDeviceManager&) = delete;
    CudaDeviceManager& operator=(const CudaDeviceManager&) = delete;

    void uploadGraph(const Graph& graph);
    Graph downloadGraph();

    void applyGraphUpdates(const std::vector<SwapPair>& swaps, const std::vector<NodeUpdate>& updates);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};


int testCuda();

#endif //PAREX_INTERFACE_H
