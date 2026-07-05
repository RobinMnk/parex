# Parex

Parex is a C++/CUDA project for fast, random-walk based expander decompositions and congestion approximators.

It provides both a single-threaded CPU implementation and a CUDA-accelerated GPU implementation.

The implementation is based on the following paper.

> Robin Münk, "A Practical Parallel Algorithm for Expander Decompositions," in *Proceedings of the 38th ACM Symposium on Parallelism in Algorithms and Architectures* (SPAA '26), 2026, doi: [10.1145/3816782.3819180](https://doi.org/10.1145/3816782.3819180)

## Algorithmic Structure

Parex can be seen as a pipeline:

```text
expander decomposition -> congestion approximator -> normalized cut
```

Each stage is useful on its own. The expander decomposition partitions the input graph into clusters with good expansion properties. The congestion approximator builds a higher-level structure from those decompositions. The normalized cut routine then uses that structure to compute normalized cut values and partitions.

At the base of the pipeline, Parex provides two implementations of expander decomposition:

| Implementation | Description |
| --- | --- |
| CPU expander decomposition | A single-threaded reference implementation of the expander decomposition algorithm. |
| CUDA expander decomposition | A GPU-accelerated implementation designed to run the decomposition more efficiently on NVIDIA GPUs. |

These implementations can be used as the backbone for:

| Routine | Description |
| --- | --- |
| Congestion approximator | Builds an approximation structure from expander decompositions. Intermediate approximators are independently useful for algorithms that need compact information about graph cuts and congestion. |
| Normalized cut | Uses the congestion approximator to compute normalized cut values and derive graph partitions according to the normalized cut objective. |

## Requirements

- CMake 3.18 or newer
- A C++20 compiler
- NVIDIA CUDA Toolkit with C++20-capable CUDA compiler support
- An NVIDIA GPU compatible with the configured CUDA architectures
- Git, if CMake needs to fetch test dependencies

GoogleTest is fetched automatically through CMake for the test targets.

## CUDA Architecture

The CUDA library currently targets these GPU architectures:

```cmake
CUDA_ARCHITECTURES "80;86"
```

This is configured in `cuda/CMakeLists.txt`. If your GPU uses a different compute capability, update that setting before building.

## Building

From the repository root:

```powershell
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release --config Release
```

## Running

The current executable runs the CUDA expander decomposition path.

```text
Parex <filename> [seed] [sweep_cut_threshold] [random_walk_threshold]
```

Arguments:

| Argument | Description |
| --- | --- |
| `filename` | Path to the input graph file. |
| `seed` | Optional random seed. |
| `sweep_cut_threshold` | Optional sparsity threshold for accepting a sweep cut. |
| `random_walk_threshold` | Optional threshold for random-walk convergence. |

Example:

```powershell
.\cmake-build-release\Parex graphs\example.mtx 6582 0.3 1e-4
```

The program prints:

- running time
- number of clusters
- number of cut edges
- number of nodes
- number of edges

## Input Format

The current graph reader expects a Matrix Market-style coordinate input.

## Configuration

Several algorithm parameters are exposed either through the CLI or through configuration structs and global defaults:

| Parameter | Description |
| --- | --- |
| `targetSparsity` / `sc_threshold` | Threshold below which a sweep cut is considered sparse enough to split a cluster. |
| `randomWalkThreshold` / `rw_threshold` | Threshold used for random-walk convergence and potential checks. |
| `initialNumSteps` | Number of random-walk steps before the first sweep cut in the CPU implementation. |
| `mainNumSteps` | Number of random-walk steps between later rounds in the CPU implementation. |
| `randSeed` | Seed for the random source used by the CPU implementation. |

The CUDA implementation also uses constants from `lib/core/definitions.h`, including the thread and warp configuration.

## Citation

If you use Parex in academic work, please cite the paper this implementation is based on.
