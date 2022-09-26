#include <algorithm>
#include "bluebild//config.h"
#include "gpu/util/gpu_runtime.hpp"
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild{

template<int BLOCK_SIZE, typename T>
__global__ void min_diff_1d_kernel(int m, const T* x, int n, const T* y, int* indices) {
  __shared__ T storage[BLOCK_SIZE];
  int minIndex = 0;
  T minValue(0);
  T yValue(0);

  for (int idxOut = threadIdx.x + blockIdx.x * BLOCK_SIZE; (idxOut / BLOCK_SIZE) * BLOCK_SIZE < n;
       idxOut += gridDim.x * BLOCK_SIZE) {
    if (idxOut < n) yValue = y[idxOut];

    for (int i = 0; i < m; i += BLOCK_SIZE) {
      // load next block of x
      int idxRead = i + threadIdx.x;
      if (idxRead < m) {
        storage[threadIdx.x] = x[idxRead];
      }
      __syncthreads();

      // loop over loaded block
      int size = BLOCK_SIZE > m - i ? m - i : BLOCK_SIZE;
      for (int j = 0; j < size; ++j) {
        // compute the absolute difference
        T diff = storage[j] - yValue;
        if (diff < 0) diff = -diff;

        // store minimum. Always store if first iteration.
        if (diff < minValue || !(i + j)) {
          minValue = diff;
          minIndex = i + j;
        }
      }
    }
    if (idxOut < n) indices[idxOut] = minIndex;
  }
}

template <typename T>
auto min_diff_1d_gpu(gpu::StreamType stream, int m, const T* x, int n, const T* y, int* indices)
    -> void {
  constexpr int maxBlocks = 65535;
  constexpr int blockSize = 128;

  dim3 block(blockSize, 1, 1);
  dim3 grid(std::min<unsigned int>(maxBlocks, (n + block.x - 1) / block.x), 1, 1);
  gpu::launch_kernel(min_diff_1d_kernel<blockSize, T>, grid, block, 0, stream, m, x, n, y, indices);
}

template auto min_diff_1d_gpu<float>(gpu::StreamType stream, int m, const float* x, int n, const float* y, int* indices) -> void;

template auto min_diff_1d_gpu<double>(gpu::StreamType stream, int m, const double* x, int n, const double* y, int* indices) -> void;

}  // namespace bluebild
