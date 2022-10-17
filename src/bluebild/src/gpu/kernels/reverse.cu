#include <algorithm>

#include "bluebild//config.h"
#include "gpu/util/gpu_runtime.hpp"
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild {

template <typename T> __global__ void reverse_1d_kernel(std::size_t n, T *x) {
  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n / 2;
       i += gridDim.x * blockDim.x) {
    T x1 = x[i];
    T x2 = x[n - 1 - i];
    x[n - 1 - i] = x1;
    x[i] = x2;
  }
}

template <typename T>
__global__ void reverse_2d_coloumns_kernel(std::size_t m, std::size_t n, T *x,
                                           std::size_t ld) {
  for (std::size_t i = blockIdx.y; i < n / 2; i += gridDim.y) {
    for (std::size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < m;
         j += gridDim.x * blockDim.x) {
      T x1 = x[i * ld + j];
      T x2 = x[(n - 1 - i) * ld + j];
      x[(n - 1 - i) * ld + j] = x1;
      x[i * ld + j] = x2;
    }
  }
}

template <typename T>
auto reverse_1_gpu(gpu::StreamType stream, std::size_t n, T *x) -> void {
  constexpr std::size_t maxBlocks = 65535;
  constexpr std::size_t blockSize = 256;

  dim3 block(blockSize, 1, 1);
  dim3 grid(std::min<std::size_t>(maxBlocks, (n + block.x - 1) / block.x), 1,
            1);
  gpu::launch_kernel(reverse_1d_kernel<T>, grid, block, 0, stream, n, x);
}

template <typename T>
auto reverse_2_gpu(gpu::StreamType stream, std::size_t m, std::size_t n, T *x,
                   std::size_t ld) -> void {
  constexpr std::size_t maxBlocks = 65535;
  constexpr std::size_t blockSize = 256;

  dim3 block(blockSize / 8, 8, 1);
  dim3 grid(std::min<std::size_t>(maxBlocks, (m + block.x - 1) / block.x),
            std::min<std::size_t>(maxBlocks, (n + block.y - 1) / block.y), 1);
  gpu::launch_kernel(reverse_2d_coloumns_kernel<T>, grid, block, 0, stream, m,
                     n, x, ld);
}

template auto reverse_1_gpu<float>(gpu::StreamType stream, std::size_t n,
                                   float *x) -> void;

template auto reverse_1_gpu<double>(gpu::StreamType stream, std::size_t n,
                                    double *x) -> void;

template auto reverse_1_gpu<gpu::ComplexType<float>>(gpu::StreamType stream,
                                                     std::size_t n,
                                                     gpu::ComplexType<float> *x)
    -> void;

template auto
reverse_1_gpu<gpu::ComplexType<double>>(gpu::StreamType stream, std::size_t n,
                                        gpu::ComplexType<double> *x) -> void;

template auto reverse_2_gpu<float>(gpu::StreamType stream, std::size_t m,
                                   std::size_t n, float *x, std::size_t ld)
    -> void;

template auto reverse_2_gpu<double>(gpu::StreamType stream, std::size_t m,
                                    std::size_t n, double *x, std::size_t ld)
    -> void;

template auto reverse_2_gpu<gpu::ComplexType<float>>(gpu::StreamType stream,
                                                     std::size_t m,
                                                     std::size_t n,
                                                     gpu::ComplexType<float> *x,
                                                     std::size_t ld) -> void;

template auto reverse_2_gpu<gpu::ComplexType<double>>(
    gpu::StreamType stream, std::size_t m, std::size_t n,
    gpu::ComplexType<double> *x, std::size_t ld) -> void;

} // namespace bluebild
