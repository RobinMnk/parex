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

int main() {
    checkBuildMode();

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
