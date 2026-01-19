#include "lib/algorithms/normalized_cut.h"
#include "lib/utils/io.h"
#include "lib/utils/timer.h"


void checkBuildMode() {
    #ifdef NDEBUG
        std::cout << "Running in RELEASE mode!" << std::endl;
    #else
        std::cout << "Running in DEBUG mode" << std::endl;
    #endif
}

void test(Graph& G) {
    CudaDeviceManager devGraph;
    devGraph.initialize(G);

//    std::vector<SwapPair> swaps;
//    std::vector<NodeUpdate> updates;
//    std::swap(G.edges[0], G.edges[1]);
//    std::swap(G.edges[54], G.edges[10]);
//    std::swap(G.edges[24], G.edges[95]);
//    swaps.emplace_back(0, 1);
//    swaps.emplace_back(54, 10);
//    swaps.emplace_back(24, 95);
//
//    devGraph.applyGraphUpdates(swaps,updates);

    Graph G2 = devGraph.downloadGraph();

    std::cout << (G.edges == G2.edges) << std::endl;
    std::cout << (G.ranges == G2.ranges) << std::endl;
    std::cout << (G.weights == G2.weights) << std::endl;
    std::cout << (G.weights.size() == G2.weights.size()) << std::endl;
    std::cout << (G.numNodes == G2.numNodes) << std::endl;
    std::cout << (G.numEdges == G2.numEdges) << std::endl;
    std::cout << (G == G2) << std::endl;
}

void test2(Graph& G) {
    CudaDeviceManager cuda;
    cuda.initialize(G);

    auto x = cuda.readRandomWalkValues();

    RandomWalk rw(G.numNodes);
    rw.setData(x);
    Partition part(&G);

    std::cout << "GPU [0] = " << x[0] << std::endl;
    std::cout << "CPU [0] = " << rw.values()[0] << "\n" << std::endl;

    cuda.iterateRandomWalk();
    rw.iterate(part, {0});

    auto y = cuda.readRandomWalkValues();
    auto z = rw.values();

    std::cout << "GPU [0] = " << y[0] << std::endl;
    std::cout << "CPU [0] = " << z[0] << "\n" << std::endl;

    for(NodeIx nix = 0; nix < G.numNodes; nix++) {
        if(std::abs(y[nix] - z[nix]) > 0.00001) {
            std::cerr << "ERROR at nix " << nix << std::endl;
        }
    }


    Graph G2 = cuda.downloadGraph();
    std::cout << (G == G2) << std::endl;
}

int main() {
    checkBuildMode();

    std::cout << "Reading input graph: ";
    std::cout.flush();
    DynamicGraph G_dyn = readDynGraph("../graphs/coPapersCiteseer.mtx");
    Graph G = G_dyn.finalize();
    std::cout << "loaded " << G.numNodes << " nodes and " << G.numEdges << " edges\nBegin Expander Decomposition" << std::endl;

    test2(G);

    return 1;

    Timer t;
    t.start();
    auto ed = expanderDecomposition(G);
    auto timeSpent = t.timeSeconds();

    std::cout << "\nFinished:  (" << timeSpent << "s)\n-> "
              << ed.numClusters() << " clusters with " << ed.getNumCutEdges() << " crossing edges" << std::endl;

//    writePartition(ed, "uk");

//    ExpanderHierarchy eh{&G};
//    eh.build();
//
//    auto timeSpent = t.timeSeconds();
//
//    std::cout << "\nFinished:  (" << timeSpent << "s)" << std::endl;
//
//    t.start();
//    NormalizedCut nc{&eh};
//    Partition kPart = nc.compute(16);
//
//    timeSpent = t.timeSeconds();
//
//    frac_t ncVal = compute_normalized_cut(kPart);
//    WARN("Checked NC:   \t" << ncVal << "\t\t[" << timeSpent << "s]");

}
