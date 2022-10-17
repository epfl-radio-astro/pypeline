#include <algorithm>

#include "bluebild//config.h"
#include "gpu/kernels/scale_matrix.hpp"
#include "gpu/util/gpu_runtime.hpp"
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild {
namespace gpu {

template <typename T>
__global__ void scale_matrix_kernel(std::size_t m, std::size_t n,
                                    const gpu::ComplexType<T> *A,
                                    std::size_t lda, const T *x,
                                    gpu::ComplexType<T> *B, std::size_t ldb) {
  for (std::size_t j = blockIdx.y; j < n; j += gridDim.y) {
    const auto valX = x[j];
    for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < m;
         i += gridDim.x * blockDim.x) {
      const auto valA = A[j * lda + i];
      B[j * ldb + i] = {valA.x * valX, valA.y * valX};
    }
  }
}

template <typename T>
auto scale_matrix(gpu::StreamType stream, std::size_t m, std::size_t n,
                  const gpu::ComplexType<T> *A, std::size_t lda, const T *x,
                  gpu::ComplexType<T> *B, std::size_t ldb) -> void {
  constexpr std::size_t blockSize = 256;
  constexpr std::size_t maxBlocks = 65535;

  dim3 block(blockSize, 1, 1);
  dim3 grid(std::min<std::size_t>(maxBlocks, (m + block.x - 1) / block.x),
            std::min<std::size_t>(maxBlocks, (n + block.y - 1) / block.y), 1);
  gpu::launch_kernel(scale_matrix_kernel<T>, grid, block, 0, stream, m, n, A,
                     lda, x, B, ldb);
}

template auto
scale_matrix<float>(gpu::StreamType stream, std::size_t m, std::size_t n,
                    const gpu::ComplexType<float> *A, std::size_t lda,
                    const float *x, gpu::ComplexType<float> *B, std::size_t ldb)
    -> void;

template auto scale_matrix<double>(gpu::StreamType stream, std::size_t m,
                                   std::size_t n,
                                   const gpu::ComplexType<double> *A,
                                   std::size_t lda, const double *x,
                                   gpu::ComplexType<double> *B, std::size_t ldb)
    -> void;

} // namespace gpu
} // namespace bluebild
