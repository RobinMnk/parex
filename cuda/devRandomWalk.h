//
// Created by robin on 19.01.2026.
//

#ifndef PAREX_DEVRANDOMWALK_H
#define PAREX_DEVRANDOMWALK_H

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

#include <curand_kernel.h>

#include <cub/cub.cuh>
#include <cassert>
#include <thrust/detail/binary_search.inl>

#include "devGraph.h"


struct NormalDistributionFunctor {
    unsigned int base_seed;

    explicit NormalDistributionFunctor(unsigned int s) : base_seed(s) {}

    __host__ __device__
    frac_t operator()(const NodeIx idx) const {
        thrust::default_random_engine rng(base_seed);
        thrust::normal_distribution<frac_t> dist;
        rng.discard(idx);
        return dist(rng);
//        return static_cast<frac_t>(idx) / 4096;
    }
};


__device__
inline uint64_t packKey(int label, float v) {
    uint32_t i = __float_as_uint(v);

    uint32_t mask = (i & 0x80000000) ? 0xffffffff : 0x80000000;
    uint32_t ordered = i ^ mask;

    uint32_t orderedLabel = static_cast<uint32_t>(label) ^ 0x80000000;

    // Shift label to high bits, place ordered float in low bits
    return (static_cast<uint64_t>(orderedLabel) << 32) | ordered;
}

// __global__
// void lazyRandomWalkKernel(
//         NodeIx numNodes,
//         const NodeIx* __restrict__ neighbors,
//         const EdgeIx* __restrict__ ranges,
//         NodeData* __restrict__ nodeData,
//         const EdgeIx* __restrict__ activeDegrees,
//         const frac_t* __restrict__ old_dist,
//         frac_t* __restrict__ dist,
//         uint64_t* __restrict__ packedKeys
// ) {
//     NodeIx i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= numNodes) return;
//
//     // early exit for inactive nodes
//     const NodeData data = nodeData[i];
//     if(data.label < 0) {
//         packedKeys[i] = packKey(data.label, 0);
//         return;
//     }; // label < 0 considered inactive
//
//     assert(data.nix == i && "nix mismatch!");
//
//     frac_t incoming_sum = 0.0f;
//     const EdgeIx rangeStart = ranges[i];
//     const EdgeIx rangeEnd = ranges[i] + data.degree;
//
//     for (NodeIx j = rangeStart; j < rangeEnd; ++j) {
//         const NodeIx neighbor = __ldg(&neighbors[j]);
//         const NodeData nbData = nodeData[neighbor];
//         const EdgeIx nbDeg = __ldg(&activeDegrees[neighbor]);
//
//         if(nbData.label == data.label && nbDeg != 0) {
//             incoming_sum += (__ldg(&old_dist[neighbor]) / static_cast<frac_t>(nbDeg));
//         }
//     }
//
//     const frac_t nodeVal = (incoming_sum * (1.0f - rw_stay)) + (old_dist[i] * rw_stay);
//
//     dist[i] = nodeVal;
//     nodeData[i].activeDegree = activeDegrees[i];
//
//     // FOR SWEEP CUT:
//     // Already Pack label and rw value into packedKeys for sorting!!
//     packedKeys[i] = packKey(data.label, nodeVal);
// }


struct LabelExtractorRW {
    __host__ __device__
    inline int operator()(const NodeData& data) const {
        return data.label;
    }
};

struct ClusterData {
    float rwSum;
    float maxPotential;
    float minPotential;
    NodeIx totalElements;
};

struct ClusterDataExtractorRW {
    __device__
    inline ClusterData operator()(const NodeData& nodeData) const {
        const float rwValue = nodeData.rwValue;
        return {rwValue, rwValue, rwValue, 1};
    }
};


struct ClusterDataReduceOp {
    __host__ __device__
    ClusterData operator()(const ClusterData& a, const ClusterData& b) const {
        return {
            a.rwSum + b.rwSum,
            a.maxPotential > b.maxPotential ? a.maxPotential : b.maxPotential,
            a.minPotential < b.minPotential ? a.minPotential : b.minPotential,
            a.totalElements + b.totalElements
        };
    }
};


struct WalkEdgeLogic {
    const NodeIx numNodes;
    const NodeIx* __restrict__ neighbors;
    // const EdgeIx* __restrict__ edgeMap;
    const EdgeIx* __restrict__ activeDegrees;
    const frac_t* __restrict__ dist;
    // const NodeData* __restrict__ nodes;

    __device__ __forceinline__
    float operator()(EdgeIx edgeIdx) const {
        // const EdgeIx revEdge = edgeMap[edgeIdx];
        // const NodeIx srcNode = neighbors[revEdge];
        //
        // const NodeIx tgtNode = neighbors[edgeIdx];
        //
        // if (__ldg(&nodes[srcNode].label) != __ldg(&nodes[tgtNode].label)) {
        //     return 0.0f;
        // }
        const NodeIx tgtNode = __ldg(&neighbors[edgeIdx]);
        if (tgtNode == numNodes+1000) {
            // inactive edge
            return 0.0f;
        }


        const EdgeIx nbDeg = __ldg(&activeDegrees[tgtNode]);

        if (nbDeg > 0) {
            return __ldg(&dist[tgtNode]) / static_cast<float>(nbDeg);
        }
        return 0.0f;
    }
};

__global__
void finalizeRandomWalk(
    NodeIx numNodes,
    const EdgeIx* __restrict__ activeDegrees,
    const frac_t* __restrict__ incoming_sums,
    frac_t* __restrict__ dist,
    NodeData* __restrict__ nodeData,
    uint64_t* __restrict__ packedKeys
) {
    NodeIx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;

    const NodeData data = nodeData[i];
    if (data.label < 0) {
        packedKeys[i] = packKey(data.label, 0);
        return;
    }

    const frac_t nodeVal = (incoming_sums[i] * (1.0f - rw_stay)) + (dist[i] * rw_stay);

    dist[i] = nodeVal;
    nodeData[i].activeDegree = activeDegrees[i];
    packedKeys[i] = packKey(data.label, nodeVal);
}



class RandomWalkManager {
    thrust::device_vector<frac_t> dist;
    thrust::device_vector<frac_t> incomingSums;

    thrust::device_vector<ClusterData> clusterSums;

    NodeIx numNodes;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    int maxLabel{0};

public:
    explicit RandomWalkManager(NodeIx n) : dist(n), incomingSums(n), clusterSums(n), numNodes(n) {
        initRandomWalk(seed);
        prepare_cub();
    }

    ~RandomWalkManager() {
        if (d_temp_storage) cudaFree(d_temp_storage);
    }

    auto& getClusterData() {
        return clusterSums;
    }

    int computeClusterData(cub::DoubleBuffer<NodeData>& partition, thrust::device_vector<int>& uniqueLabels) {
        NodeData* partitionPtr = partition.Current();
        auto label_iter = thrust::make_transform_iterator(partitionPtr, LabelExtractorRW());
        auto value_iter = thrust::make_transform_iterator(partitionPtr, ClusterDataExtractorRW());

        // find ClusterData for each cluster (to compute potential and average of each)
        auto end_iters = thrust::reduce_by_key(
            thrust::device,
            label_iter,
            label_iter + numNodes,
            value_iter,
            uniqueLabels.begin(),
            clusterSums.begin(),
            thrust::equal_to<int>(),
            ClusterDataReduceOp()
        );


        int numUnique = end_iters.second - clusterSums.begin();

        thrust::sort_by_key(
            uniqueLabels.begin(),
            uniqueLabels.begin() + numUnique,
            clusterSums.begin()
        );

        thrust::copy_n(uniqueLabels.begin() + numUnique - 1, 1, &maxLabel);

        // printf("There are %d clusters and the maximum label is %d\n", numUnique, maxLabel);

        return numUnique;
    }


    void print(int numUnique, thrust::device_vector<int>& uniqueLabels) {
        std::vector<ClusterData> cst(numUnique);
        thrust::copy(clusterSums.begin(), clusterSums.begin() + numUnique, cst.begin());
        std::vector<int> lbl(numUnique);
        thrust::copy(uniqueLabels.begin(), uniqueLabels.begin() + numUnique, lbl.begin());
        for (int i = 0; i < numUnique; i++) {
            const float clusterPotential = cst[i].maxPotential - cst[i].minPotential;
            const float average = cst[i].rwSum / static_cast<float>(cst[i].totalElements);
            printf("%d:  Cluster %d [average = %f, potential = %f, numElements = %d]\n", i, lbl[i], average, clusterPotential, cst[i].totalElements);
        }
        fflush(stdout);
    }

    int recenterAndDeactivateClusters(cub::DoubleBuffer<NodeData>& partition, thrust::device_vector<int>& uniqueLabels) {
        int numUnique = computeClusterData(partition, uniqueLabels);

        cudaDeviceSynchronize();


        // printf("Before:\n");
        // print(numUnique, uniqueLabels);


        // subtract average from each (active) node
        const ClusterData* clusterDataPtr = thrust::raw_pointer_cast(clusterSums.data());
        const int* uniqueLabelsPtr = thrust::raw_pointer_cast(uniqueLabels.data());

        frac_t* distPtr = thrust::raw_pointer_cast(dist.data());

        thrust::for_each_n(
            thrust::device,
            partition.Current(),
            numNodes,
            [clusterDataPtr, uniqueLabelsPtr, distPtr, numUnique] __device__ (NodeData& data) {
                const int label = data.label;

                if (data.label < 0) {
                    // printf("node %d with label %d returns because label is negative\n", data.nix, label);
                    return;
                }; // cluster already inactive

                const int* it = thrust::lower_bound(
                    thrust::seq,
                    uniqueLabelsPtr,
                    uniqueLabelsPtr + numUnique,
                    label
                );
                int correspondingSweepCutIndex = static_cast<int>(it - uniqueLabelsPtr);

                if (uniqueLabelsPtr[correspondingSweepCutIndex] != label) {
                    printf("ERROR: label mismatch!! For nix = %d:\t%d != %d\n", data.nix, label, uniqueLabelsPtr[correspondingSweepCutIndex]);
                }

                const ClusterData cd = clusterDataPtr[correspondingSweepCutIndex];

                // if (cd.totalElements < 2) {
                //     // printf("node %d with label %d returns because numElements = %d. Note that totalClusters = %d\n", data.nix, label, cd.totalElements, numUnique);
                //     return;
                // };

                const float clusterPotential = cd.maxPotential - cd.minPotential;

                if (clusterPotential < rw_threshold || cd.totalElements < 2) {
                    // this cluster should be deactivated
                    int smallestLabel = uniqueLabelsPtr[0];
                    // printf("Deactivating cluster: %d -> %d \t smallest: %d\n", label, smallestLabel - data.label - 1, smallestLabel);
                    data.label = smallestLabel - data.label - 1;
                } else {
                    data.label = correspondingSweepCutIndex;

                    const float average = cd.rwSum / static_cast<double>(cd.totalElements);
                    distPtr[data.nix] -= average; // TODO: write this diff to any unused field within NodeData! (save random write)
                    data.rwValue -= average; // TODO: this line is just for debugging
                }

                // printf("moved node %d from cluster %d to cluster %d\n", data.nix, label, data.label);

                // printf("node %d with label %d, old rwValue = %f and offset = %d loaded this cluster data (%d): [average = %f, potential = %f, numElements = %d]\n", data.nix, label, rw, data.offsetInCluster, correspondingSweepCutIndex, average, cd.maxPotential - cd.minPotential, cd.totalElements);

            }
        );

        // printf("After\n");
        // int x = computeClusterData(partition, uniqueLabels);
        // // assert(x-1 == maxLabel);
        // print(numUnique, uniqueLabels);
        // printf("\n");

        return numUnique;
    }

    void stepFast(GraphManager& gm,
          cub::DoubleBuffer<NodeData>& partition,
          cub::DoubleBuffer<uint64_t>& packedKeys,
          thrust::device_vector<EdgeIx>& activeDegrees
    ) {
        const EdgeIx* edgeMapPtr = thrust::raw_pointer_cast(gm.getEdgeMap().data());
        const EdgeIx* rangesPtr = thrust::raw_pointer_cast(gm.getRanges().data());
        const NodeIx* neighborsPtr = thrust::raw_pointer_cast(gm.getNeighbors().data());
        const EdgeIx* activeDegsPtr = thrust::raw_pointer_cast(activeDegrees.data());
        frac_t* distPtr = thrust::raw_pointer_cast(dist.data());
        frac_t* incomingSumsPtr = thrust::raw_pointer_cast(incomingSums.data());

        thrust::counting_iterator<EdgeIx> edgeIndexIter(0);
        WalkEdgeLogic functor{numNodes, neighborsPtr, activeDegsPtr, distPtr};

        auto transIter = thrust::make_transform_iterator(edgeIndexIter, functor);

        // perform the reduction
        cub::DeviceSegmentedReduce::Sum(
            d_temp_storage, temp_storage_bytes,
            transIter, incomingSumsPtr,
            numNodes, rangesPtr, rangesPtr + 1
        );

        // Finalize
        int gridSize = (numNodes + threads - 1) / threads;

        finalizeRandomWalk<<<gridSize, threads, 0, nullptr>>>(
            numNodes,
            activeDegsPtr,
            incomingSumsPtr,
            distPtr,
            partition.Current(),
            packedKeys.Current()
        );
    }



    // void step(GraphManager& gm,
    //       cub::DoubleBuffer<NodeData>& partition,
    //       cub::DoubleBuffer<uint64_t>& packedKeys,
    //       thrust::device_vector<EdgeIx>& activeDegrees,
    //       cudaStream_t stream = nullptr
    // ) {
    //
    //     // BEFORE:
    //
    //     // recenter clusters -> find potential -> deactivate if necessary!!
    //
    //     old_dist.swap(dist);
    //
    //     unsigned int blocksPerGrid = (numNodes + threads - 1) / threads;
    //
    //     lazyRandomWalkKernel<<<blocksPerGrid, threads, 0, stream>>>(
    //             numNodes,
    //             thrust::raw_pointer_cast(gm.getNeighbors().data()),
    //             thrust::raw_pointer_cast(gm.getRanges().data()),
    //             partition.Current(),
    //             thrust::raw_pointer_cast(activeDegrees.data()),
    //             thrust::raw_pointer_cast(old_dist.data()),
    //             thrust::raw_pointer_cast(dist.data()),
    //             packedKeys.Current()
    //     );
    // }

    int getMaxLabel() const {
        return maxLabel;
    }


//    void newStep(GraphManager& gm,
//              cub::DoubleBuffer<NodeData> partition,
//              cub::DoubleBuffer<uint64_t> packedKeys,
//              cudaStream_t stream = nullptr
//    ) {
//
//        thrust::transform(thrust::cuda::par.on(stream),
//                          old_dist.begin(), old_dist.end(),
//                          gr.active_degrees,
//                          node_val.begin(),
//            [move_weight] __device__ (frac_t d, EdgeIx deg) {
//                return (deg > 0) ? (d / static_cast<frac_t>(deg)) * move_weight : 0.0f;
//            }
//        );
//
//    }



//    void stepSlow(GraphManager& gm, cudaStream_t stream = nullptr) {
//        old_dist.swap(dist);
//
//        const frac_t stay_weight = rw_stay;
//        const frac_t move_weight = 1.0f - rw_stay;
//
//        // node_val = (old_dist / degree) * move_weight
//        thrust::transform(thrust::cuda::par.on(stream),
//                          old_dist.begin(), old_dist.end(),
//                          gr.active_degrees,
//                          node_val.begin(),
//            [move_weight] __device__ (frac_t d, EdgeIx deg) {
//                return (deg > 0) ? (d / static_cast<frac_t>(deg)) * move_weight : 0.0f;
//            }
//        );
//
//        frac_t* raw_node_vals = thrust::raw_pointer_cast(node_val.data());
//        frac_t* raw_dist_out = thrust::raw_pointer_cast(dist.data());
//        NodeIx* raw_ranges = thrust::raw_pointer_cast(gm.getRanges().data());
//
//        auto v_mapped_iter = thrust::make_permutation_iterator(raw_node_vals, gr.neighbors);
//
//        // dist = sum_neighbors node_val
//        cub::DeviceSegmentedReduce::Sum(
//                d_temp_storage,
//                temp_storage_bytes,
//                v_mapped_iter,
//                raw_dist_out,
//                static_cast<int>(numNodes),
//                raw_ranges,
//                raw_ranges + 1,
//                stream
//        );
//
//        // dist += (1 - rw_stay) * old_dist
//        thrust::transform(thrust::cuda::par.on(stream),
//                          dist.begin(), dist.end(),
//                          old_dist.begin(),
//                          dist.begin(),
//            [stay_weight] __device__ (frac_t summed_neighbors, frac_t self_old) {
//                return summed_neighbors + (self_old * stay_weight);
//            }
//        );
//    }

    [[nodiscard]] const thrust::device_vector<frac_t>& randomWalkValues() const {
        return dist;
    }

    std::vector<frac_t> valuesToCPU() {
        std::vector<frac_t> rwVals(numNodes);
        thrust::copy(dist.begin(), dist.end(), rwVals.begin());
        return rwVals;
    }

private:
    void prepare_cub() {
        WalkEdgeLogic dryRunFunctor{numNodes, nullptr, nullptr, nullptr};
        auto dryRunIter = thrust::make_transform_iterator(thrust::make_counting_iterator<EdgeIx>(0), dryRunFunctor);

        EdgeIx* nullOffsets = static_cast<EdgeIx*>(nullptr);
        frac_t* nullOutput  = static_cast<frac_t*>(nullptr);

        // Dry run to get temp_storage_bytes
        cudaError_t err = cub::DeviceSegmentedReduce::Sum(
            nullptr, temp_storage_bytes,
            dryRunIter, nullOutput,
            static_cast<int>(numNodes), nullOffsets, nullOffsets
        );

        if (err != cudaSuccess) {
            printf("CUB Error: %s\n", cudaGetErrorString(err));
        }

        cudaMalloc(&d_temp_storage, temp_storage_bytes);
    }

    void initRandomWalk(unsigned int s) {
        thrust::transform(thrust::make_counting_iterator<NodeIx>(0),
                          thrust::make_counting_iterator(numNodes),
                          dist.begin(),
                          NormalDistributionFunctor(s)
        );
    }
};

#endif //PAREX_DEVRANDOMWALK_H
