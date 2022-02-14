#include <algorithm>
#include "bluebild//config.h"
#include "gpu/util/gpu_runtime.hpp"
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild{

template<typename T>
__global__ void reverse_1d_kernel(int n, T* x) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n / 2; i += gridDim.x * blockDim.x) {
    T x1 = x[i];
    T x2 = x[n - 1 - i];
    x[n - 1 - i] = x1;
    x[i] = x2;
  }
}

template <typename T>
__global__ void reverse_2d_coloumns_kernel(int m, int n, T* x, int ld) {
  for (int i = blockIdx.y; i < n / 2; i += gridDim.y) {
    for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < m; j += gridDim.x * blockDim.x) {
      T x1 = x[i * ld + j];
      T x2 = x[(n - 1 - i) * ld + j];
      x[(n - 1 - i) * ld + j] = x1;
      x[i * ld + j] = x2;
    }
  }
}

template <typename T>
auto reverse_1_gpu(gpu::StreamType stream, int n, T* x) -> void {
    constexpr int maxBlocks = 65535;
    constexpr int blockSize = 256;

    dim3 block(blockSize, 1, 1);
    dim3 grid(std::min<unsigned int>(maxBlocks, (n + block.x - 1) / block.x), 1, 1);
    gpu::launch_kernel(reverse_1d_kernel<T>, grid, block, 0, stream, n, x);
}

template <typename T>
auto reverse_2_gpu(gpu::StreamType stream, int m, int n, T* x, int ld) -> void {
  constexpr int maxBlocks = 65535;
  constexpr int blockSize = 256;

  dim3 block{blockSize / 8, 8, 1};
  dim3 grid{std::min<unsigned int>(maxBlocks, (m + block.x - 1) / block.x),
            std::min<unsigned int>(maxBlocks, (n + block.y - 1) / block.y), 1};
  gpu::launch_kernel(reverse_2d_coloumns_kernel<T>, grid, block, 0, stream, m, n, x, ld);
}

template auto reverse_1_gpu<float>(gpu::StreamType stream, int n, float* x) -> void;

template auto reverse_1_gpu<double>(gpu::StreamType stream, int n, double* x) -> void;

template auto reverse_1_gpu<cuComplex>(gpu::StreamType stream, int n, cuComplex* x) -> void;

template auto reverse_1_gpu<cuDoubleComplex>(gpu::StreamType stream, int n, cuDoubleComplex* x)
    -> void;

template auto reverse_2_gpu<float>(gpu::StreamType stream, int m, int n, float* x, int ld) -> void;

template auto reverse_2_gpu<double>(gpu::StreamType stream, int m, int n, double* x, int ld) -> void;

template auto reverse_2_gpu<cuComplex>(gpu::StreamType stream, int m, int n, cuComplex* x, int ld) -> void;

template auto reverse_2_gpu<cuDoubleComplex>(gpu::StreamType stream, int m, int n, cuDoubleComplex* x, int ld)
    -> void;

}  // namespace bluebild
