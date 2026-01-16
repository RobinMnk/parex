//
// Created by robin on 13.03.2025.
//

#ifndef RCUT_RANDOM_WALK_H
#define RCUT_RANDOM_WALK_H

#include "partition.h"
#include <valarray>
#include <random>

const frac_t rw_stay = 0.1;
std::mt19937 random_source{0};

class RandomWalk {
    std::valarray<frac_t> m_distribution;
    std::valarray<frac_t> m_old_distribution;
//    std::vector<frac_t> val_prec;

public:
    explicit RandomWalk (size_t n) : m_distribution(n), m_old_distribution(n) {
        init_distribution();
    }

    void init_distribution();

    frac_t resetCluster(const Cluster& cluster);

    void iterate(const Partition& part, const std::vector<NodeIx>& active);

    const std::valarray<frac_t>& values() {
        return m_distribution;
    }

    frac_t potential(const Cluster& cluster);

    void recenterClusterF(const Cluster& cluster);

    frac_t recenterCluster(const Cluster& cluster);

    frac_t averageOf(const Cluster& cluster);
};

void RandomWalk::iterate(const Partition& part, const std::vector<NodeIx>& active) {
    std::swap(m_old_distribution, m_distribution);
    m_distribution = 0;
    frac_t val;

    for(NodeIx clusterId: active) {
        const Cluster& cluster = part.getCluster(clusterId);
        for(const ClusterVertex& cv: cluster) {
            val = m_old_distribution[cv.nix] / static_cast<frac_t>(cv.internalDegree);
            for (auto it = cluster.edgeBegin(cv); it != cluster.edgeEnd(cv); ++it) {
                NodeIx nb = *it;
                m_distribution[nb] += val;
            }
        }
    }

    m_distribution *= (1.0f - rw_stay);
    m_distribution += rw_stay * m_old_distribution;
}

void RandomWalk::init_distribution() {
    std::normal_distribution<frac_t> normal_distribution;
    frac_t avg = 0.0;

    for (size_t i = 0; i < m_distribution.size(); i++) {
        // generate a normal distributed number
        auto x = normal_distribution(random_source);
        m_distribution[i] = x;

        // use welford mean algorithm to compute the average
        avg += (x - avg) / (static_cast<frac_t >(i) + 1.0f);
    }

    m_distribution -= avg;
}

frac_t RandomWalk::resetCluster(const Cluster& cluster) {
    std::normal_distribution<frac_t> normal_distribution;
    frac_t avg = 0.0;
    int i = 1;

    for (const ClusterVertex& cn: cluster) {
        auto x = normal_distribution(random_source);
        m_distribution[cn.nix] = x;
        avg += (x - avg) / static_cast<frac_t >(i);
        i++;
    }

    frac_t min = std::numeric_limits<frac_t>::max();
    frac_t max = std::numeric_limits<frac_t>::lowest();

    for (const ClusterVertex& cn: cluster) {
        m_distribution[cn.nix] -= avg;
        if(m_distribution[cn.nix] < min) min = m_distribution[cn.nix];
        if(m_distribution[cn.nix] > max) max = m_distribution[cn.nix];
    }

    return max - min;
}


frac_t RandomWalk::averageOf(const Cluster& cluster) {
    frac_t avg = 0.0;
    int i = 1;

    for(const ClusterVertex& cv: cluster) {
        auto x = m_distribution[cv.nix];
        avg += (x - avg) / static_cast<frac_t >(i);
        i++;
    }

    return avg;
}

void RandomWalk::recenterClusterF(const Cluster& cluster) {
    frac_t avg = 0.0;
    int i = 1;

    for (const ClusterVertex& cn: cluster) {
        auto x = m_distribution[cn.nix];
        avg += (x - avg) / static_cast<frac_t >(i);
        i++;
    }

    for (const ClusterVertex& cn: cluster) {
        m_distribution[cn.nix] -= avg;
    }
}

frac_t RandomWalk::potential(const Cluster& cluster) {
    frac_t min = std::numeric_limits<frac_t>::max();
    frac_t max = std::numeric_limits<frac_t>::lowest();

    for (const ClusterVertex& cn: cluster) {
        min = std::min(min, m_distribution[cn.nix]);
        max = std::max(max, m_distribution[cn.nix]);
    }

    return max - min;
}

/**
 * Returns the potential of the cluster
 */
frac_t RandomWalk::recenterCluster(const Cluster& cluster) {
    frac_t avg = 0.0;
    int i = 1;

    for (const ClusterVertex& cn: cluster) {
        auto x = m_distribution[cn.nix];
        avg += (x - avg) / static_cast<frac_t >(i);
        i++;
    }

    frac_t min = std::numeric_limits<frac_t>::max();
    frac_t max = std::numeric_limits<frac_t>::lowest();

    for (const ClusterVertex& cn: cluster) {
        m_distribution[cn.nix] -= avg;
        if(m_distribution[cn.nix] < min) min = m_distribution[cn.nix];
        if(m_distribution[cn.nix] > max) max = m_distribution[cn.nix];
    }

    return max - min;
}

#endif //RCUT_RANDOM_WALK_H
