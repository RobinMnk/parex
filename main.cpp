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

void test() {
    std::cout << "Reading input graph: ";
    std::cout.flush();
    DynamicGraph G_dyn = readDynGraph("../graphs/uk.mtx");
    Graph G = G_dyn.finalize();
    std::cout << "loaded " << G.numNodes << " nodes and " << G.numEdges << " edges\nBegin Expander Decomposition" << std::endl;

    CudaDeviceManager devGraph;
    devGraph.uploadGraph(G);

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

int main() {
    checkBuildMode();
    test();

    return 1;

    std::cout << "Reading input graph: ";
    std::cout.flush();
    DynamicGraph G_dyn = readDynGraph("../graphs/uk.mtx");
    Graph G = G_dyn.finalize();
    std::cout << "loaded " << G.numNodes << " nodes and " << G.numEdges << " edges\nBegin Expander Decomposition" << std::endl;

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
