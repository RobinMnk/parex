#include "lib/algorithms/normalized_cut.h"
#include "lib/utils/graph_io.h"
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
    DynamicGraph G_dyn = readDynGraph("../graphs/coPapersCiteseer.mtx");
    Graph G = G_dyn.finalize();
    std::cout << "loaded " << G.numNodes << " nodes and " << G.numEdges << " edges\nBegin Expander Decomposition" << std::endl;

    Timer t2;
    t2.start();
    CudaDeviceManager cuda;
    cuda.initialize(G);
    cuda.expanderDecomposition();
    // auto pt = cuda.downloadLabels();
    auto tm = t2.timeSeconds();
    // int numClusters = cuda.getNumClusters();
    FinalPartition fpt = cuda.getFinalPartition();
    printf("Terminated after %fs\t-> %d clusters.\n", tm, fpt.numClusters);

    t2.start();
    Partition part(&G);
    std::vector<NodeIx> mod;
    part.splitByIndices<false, false>(0, fpt.clusterIds, fpt.numClusters, mod);

    printf(" -> Partition has %d clusters and cuts %d edges\t[computed in %fs]", part.numClusters(), part.getNumCutEdges(), t2.timeSeconds());


    return 0;

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

void test2(Graph& G) {
    CudaDeviceManager cuda;
    cuda.initialize(G);

    auto x = cuda.readRandomWalkValues();

    RandomWalk rw(G.numNodes);
    rw.setData(x);
    Partition part(&G);

    std::cout << "GPU [0] = " << x[0] << std::endl;
    std::cout << "CPU [0] = " << rw.values()[0] << "\n" << std::endl;

    Timer t;
    t.start();
    rw.iterate(part, {0});
    auto timeCPU = t.timeNanos();
    std::cout << "CPU Random Walk time: " << timeCPU << "ns" << std::endl;

    t.start();
    cuda.iterateRandomWalk();
    auto timeGPU = t.timeNanos();
    std::cout << "GPU Random Walk time: " << timeGPU << "ns" << std::endl;
    std::cout << "  -> GPU faster by factor " << timeCPU / static_cast<float>(timeGPU) << "\n" << std::endl;

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