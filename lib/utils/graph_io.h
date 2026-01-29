//
// Created by robin on 13.03.2025.
//

#ifndef RCUT_IO_H
#define RCUT_IO_H

#include "core/graph.h"
#include <fstream>
#include <iostream>
#include <sstream>

Graph readGraph(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Unable to open file." << std::endl;
        exit(1);
    }

    NodeIx numNodes, fx, tx;
    EdgeIx numEdges;
    infile >> numNodes >> numEdges;

    std::vector<NodeIx> edges(2 * numEdges);
    std::vector<EdgeIx> ranges(numNodes + 1);

    EdgeIx eix;
    NodeIx nix = 0;
    for(eix = 0; eix < 2 * numEdges; eix++) {
        infile >> fx >> tx;
        edges.at(eix) = tx;
        if(fx > nix) ranges.at(++nix) = eix;
    }
    infile.close();
    ranges.at(++nix) = eix;

    std::vector<EdgeIx> map(2 * numEdges);
    std::vector<EdgeIx> tmp(ranges);
    for(eix = 0; eix < 2 * numEdges; eix++) {
        tx = edges.at(eix);
        EdgeIx pix = tmp.at(tx)++;
        map.at(eix) = pix;
        map.at(pix) = eix;
    }

    return {std::move(edges), std::move(ranges),std::move(map), numNodes, numEdges};
}

DynamicGraph readDynGraph(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Unable to open file." << std::endl;
        exit(1);
    }

    std::string line;
    if (!std::getline(infile, line)) {
        return {}; // Or throw an error
    }

    while(line.starts_with("%")) {
        if (!std::getline(infile, line)) {
            return {};
        }
    }

    NodeIx numNodes, fx, tx;
    EdgeIx numEdges;

    std::stringstream ss(line);
    NodeIx garbage;
    ss >> garbage >> numNodes >> numEdges;

    std::vector<Edge> edges(numEdges);

    EdgeIx eix;
    for(eix = 0; eix < numEdges; eix++) {
        infile >> fx >> tx;
        edges.at(eix) = {fx-1, tx-1};
    }
    infile.close();

    return {std::move(edges), numNodes, numEdges};
}

void printGraph(const DynamicGraph& G) {
    std::ofstream f;
    f.open ("../graphs/G_dyn.txt");
    f << G.numNodes << " " << G.numEdges << "\n";
    for(auto& [fx, tx]: G.edges) {
        if(fx == 0 && tx == 0) break;
        if(fx == tx) printf("Self-loops not allowed!!");
        f << static_cast<int>(fx) << " " << static_cast<int>(tx) << "\n";
        f << static_cast<int>(tx) << " " << static_cast<int>(fx) << "\n";
    }
    f.close();
}

void printGraph(const Graph& G) {
    std::ofstream f;
    f.open ("../graphs/G.txt");
    f << static_cast<int>(G.numNodes) << " " << static_cast<int>(G.numEdges) << "\n";
    for(NodeIx nix = 0; nix < G.numNodes; nix++) {
        ITER_EIX(G, nix, eix) {
            NodeIx tx = G.edges.at(eix);
            if(nix == tx) printf("Self-loops not allowed!!\t%d <-> %d\n", nix, tx);
            f << static_cast<int>(nix) << " " << static_cast<int>(tx) << "\n";
        }
    }
    f.close();
}


#endif //RCUT_IO_H
