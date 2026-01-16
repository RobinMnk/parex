//
// Created by robin on 15.05.25.
//

#ifndef RCUT_PARTITION_WRITER_H
#define RCUT_PARTITION_WRITER_H

#include "../core/partition.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

void writeCluster(const Cluster& cluster, const std::string& filename) {
    std::ofstream f;
    f.open(filename);
    f << static_cast<int>(cluster.size()) << " " << static_cast<int>(cluster.internalVolume) << "\n";

    std::unordered_map<NodeIx, size_t> lookup;
    size_t ix{0};
    for (const ClusterVertex& cv: cluster) {
        lookup[cv.nix] = ix++;
    }

    for (const ClusterVertex& cv: cluster) {
        for (auto it = cluster.edgeBegin(cv); it != cluster.edgeEnd(cv); ++it) {
            f << static_cast<int>(lookup[cv.nix]) << " " << static_cast<int>(lookup[*it]) << "\n";
        }
    }
    f.close();
}

void writePartition(const Partition& part, std::string_view dir) {
    size_t ix{0};
    std::filesystem::path base = std::filesystem::path("../graphs") / dir;

    // Ensure the directory exists
    std::error_code ec;  // Optional: suppress exceptions
    std::filesystem::create_directories(base, ec);
    if (ec) {
        std::cerr << "Failed to create directory: " << base << std::endl;
        return;
    }

    for (const Cluster& cl : part) {
        if(cl.size() == 1) continue;
        std::filesystem::path filename = base / ("part_" + std::to_string(ix++) + ".gr");
        writeCluster(cl, filename.string());
        if(ix > 50) {
            std::cerr << "Refusing to print more than 50 clusters." << std::endl;
            return;
        }
    }
}

#endif //RCUT_PARTITION_WRITER_H
