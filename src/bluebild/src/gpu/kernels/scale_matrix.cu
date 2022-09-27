#include <algorithm>
#include "bluebild//config.h"
#include "gpu/util/gpu_runtime.hpp"
#include "gpu/util/gpu_runtime_api.hpp"
#include "gpu/kernels/scale_matrix.hpp"

namespace bluebild {
namespace gpu {

template <typename T>
__global__ void scale_matrix_kernel(int m, int n, const gpu::ComplexType<T> *A,
                                    int lda, const T *x, gpu::ComplexType<T> *B,
                                    int ldb) {
  for (int j = blockIdx.y; j < n; j += gridDim.y) {
    const auto valX = x[j];
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < m;
         i += gridDim.x * blockDim.x) {
      const auto valA = A[j * lda + i];
      B[j * ldb + i] = {valA.x * valX, valA.y * valX};
    }
  }
}

template <typename T>
auto scale_matrix(gpu::StreamType stream, int m, int n,
                  const gpu::ComplexType<T> *A, int lda, const T *x,
                  gpu::ComplexType<T> *B, int ldb) -> void {
  constexpr int blockSize = 256;
  constexpr int maxBlocks = 65535;

  dim3 block(blockSize, 1, 1);
  dim3 grid(std::min<unsigned int>(maxBlocks, (m + block.x - 1) / block.x),
            std::min<unsigned int>(maxBlocks, (n + block.y - 1) / block.y), 1);
  gpu::launch_kernel(scale_matrix_kernel<T>, grid, block, 0, stream, m, n, A, lda, x, B, ldb);
}

template auto scale_matrix<float>(gpu::StreamType stream, int m, int n,
                                  const gpu::ComplexType<float> *A, int lda,
                                  const float *x, gpu::ComplexType<float> *B,
                                  int ldb) -> void;

template auto scale_matrix<double>(gpu::StreamType stream, int m, int n,
                                   const gpu::ComplexType<double> *A, int lda,
                                   const double *x, gpu::ComplexType<double> *B,
                                   int ldb) -> void;

} // namespace gpu
} // namespace bluebild
