# GPU_Programming

Clean, commented CUDA examples showing basic GPU programming patterns and performance comparisons.

This repository is a compact learning playground for CUDA programming. It contains a set of small, focused examples that demonstrate common GPU programming techniques: element-wise kernels, matrix multiplication (naive and tiled/shared-memory), tiling, and a small multi-layer perceptron (MLP) trained on MNIST implemented with explicit CUDA kernels.

## Repository structure

- `add.cu` — root example demonstrating a large vector addition using a CUDA kernel and a CPU reference for timing comparison.
- `vector_adding/add.cu` — similar vector-add example placed in a separate folder.
- `broadcastingStyleOp.cu` — an example (single-file) that demonstrates broadcasting-style operations on the GPU.
- `matrix_multiplication/matmul.cu` — naive GPU matrix multiplication and a CPU reference.
- `Tiling_for_matrix_multiplication/tiled_matmul.cu` — tiled/shared-memory matrix multiplication example with timing comparisons between naive GPU, tiled GPU, and CPU implementations. The folder also includes an `images/` directory with screenshots.
- `MLP_for_MNIST/` — a compact MLP implemented with handwritten CUDA kernels (forward, backward, parameter updates, softmax, cross-entropy). This example expects `mnist_train.csv` and `mnist_test.csv` in the folder and demonstrates training and evaluation loops.

## Prerequisites

- NVIDIA GPU with a CUDA-capable driver
- CUDA toolkit (nvcc) installed and available on `PATH`
- Standard build tools (gcc, make) if you wrap builds into scripts

Note: `MLP_for_MNIST` requires the CSV datasets (`mnist_train.csv`, `mnist_test.csv`) present in that folder — they are included in the repository snapshot.

## Build & run (examples)

General pattern (run from repository root or the example's directory):

```bash
nvcc -O2 path/to/example.cu -o example_binary
./example_binary
```

Concrete examples:

- Vector add (root):

```bash
nvcc -O2 add.cu -o add
./add
```

- Vector add (folder):

```bash
cd vector_adding
nvcc -O2 add.cu -o add
./add
```

- Naive matrix multiplication:

```bash
cd matrix_multiplication
nvcc -O2 matmul.cu -o matmul
./matmul
```

- Tiled (shared-memory) matrix multiplication:

```bash
cd Tiling_for_matrix_multiplication
nvcc -O2 tiled_matmul.cu -o tiled_matmul
./tiled_matmul
```

- MLP for MNIST (trainer):

```bash
cd MLP_for_MNIST
nvcc -O2 mnist.cu -o mnist
./mnist
```

Expected outputs

- Most examples print GPU timing (measured with CUDA events) and CPU timing (measured with `std::chrono`) along with a few sample results for quick correctness checks. For example, the matrix multiplication examples print times like:

```
#### GPU Time :  X.XXXXX ms ####
#### GPU Tiled Time :  Y.YYYYY ms ####
#### CPU Time :  Z.ZZZZZ ms ####
```

And when inputs are all ones, each matrix product entry prints as `N` (matrix size).

## Notes and suggestions

- The examples often use compile-time constants (e.g., `#define N 256`) and static host arrays. To experiment with different sizes or larger matrices, refactor to dynamic allocation and/or pass parameters via command-line arguments.
- The tiled matrix multiplication demonstrates the performance benefits of shared memory. Try varying `TILE_WIDTH` (power-of-two values: 8, 16, 32) to observe how different GPUs respond.
- The MLP implementation is educational: it implements forward, backward, softmax and parameter updates manually. For real training workloads, consider using established frameworks or cuBLAS/cuDNN primitives.
- Add basic error checks after CUDA API calls (e.g., `gpuErrchk` or checking return values) for more robust debugging and clearer diagnostics.

## Troubleshooting

- nvcc not found: install the CUDA toolkit and ensure `nvcc` is on your PATH.
- Runtime errors or OOM: lower the problem size (e.g., `N`) or run on a GPU with more memory.
- Incorrect results: enable CUDA error checks and compare GPU output to CPU reference element-wise.

## Next steps (optional improvements)

- Parameterize examples to accept `N`, block size, and tile width as command-line arguments.
- Add a small test harness that compiles and runs each example with a golden-check for correctness.
- Add a script to run benchmarks multiple times and produce mean/std timing reports.
