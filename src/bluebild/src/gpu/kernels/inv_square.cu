#include <algorithm>

#include "bluebild//config.h"
#include "gpu/util/gpu_runtime.hpp"
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild {

template <typename T>
__global__ void inv_square_1d_kernel(std::size_t n, T *x) {
  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
       i += gridDim.x * blockDim.x) {
    T val = x[i];
    x[i] = T(1) / (val * val);
  }
}

template <typename T>
auto inv_square_1d_gpu(gpu::StreamType stream, std::size_t n, T *x) -> void {
  constexpr std::size_t maxBlocks = 65535;
  constexpr std::size_t blockSize = 256;

  dim3 block(blockSize, 1, 1);
  dim3 grid(std::min<std::size_t>(maxBlocks, (n + block.x - 1) / block.x), 1,
            1);
  gpu::launch_kernel(inv_square_1d_kernel<T>, grid, block, 0, stream, n, x);
}

template auto inv_square_1d_gpu<float>(gpu::StreamType stream, std::size_t n,
                                       float *x) -> void;

template auto inv_square_1d_gpu<double>(gpu::StreamType stream, std::size_t n,
                                        double *x) -> void;

} // namespace bluebild
