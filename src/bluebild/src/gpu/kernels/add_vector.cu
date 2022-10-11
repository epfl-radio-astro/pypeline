#include <algorithm>
#include "bluebild//config.h"
#include "gpu/util/gpu_runtime.hpp"
#include "gpu/util/gpu_runtime_api.hpp"
#include "gpu/kernels/add_vector.hpp"

namespace bluebild {
namespace gpu {

template <typename T>
__global__ void
add_vector_kernel(int n, const gpu::ComplexType<T> *__restrict__ a, T *b) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
       i += gridDim.x * blockDim.x) {
    b[i] += a[i].x;
  }
}

template <typename T>
auto add_vector(gpu::StreamType stream, int n, const gpu::ComplexType<T> *a,
                T *b) -> void {
  constexpr int blockSize = 256;
  constexpr int maxBlocks = 65535;

  dim3 block(blockSize, 1, 1);
  dim3 grid(std::min<unsigned int>(maxBlocks, (n + block.x - 1) / block.x), 1,
            1);
  gpu::launch_kernel(add_vector_kernel<T>, grid, block, 0, stream, n, a, b);
}

template auto add_vector<float>(gpu::StreamType stream, int n,
                                const gpu::ComplexType<float> *a, float *b)
    -> void;

template auto add_vector<double>(gpu::StreamType stream, int n,
                                 const gpu::ComplexType<double> *a, double *b)
    -> void;

} // namespace gpu
} // namespace bluebild
