#include <algorithm>
#include "bluebild//config.h"
#include "gpu/util/gpu_runtime.hpp"
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild{

template<typename T>
__global__ void inv_square_1d_kernel(int n, T* x) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    T val = x[i];
    x[i] = T(1) / (val * val);
  }
}


template <typename T>
auto inv_square_1d_gpu(gpu::StreamType stream, int n, T* x) -> void {
    constexpr int maxBlocks = 65535;
    constexpr int blockSize = 256;

    dim3 block(blockSize, 1, 1);
    dim3 grid(std::min<unsigned int>(maxBlocks, (n + block.x - 1) / block.x), 1, 1);
    gpu::launch_kernel(inv_square_1d_kernel<T>, grid, block, 0, stream, n, x);
}


template auto inv_square_1d_gpu<float>(gpu::StreamType stream, int n, float* x) -> void;

template auto inv_square_1d_gpu<double>(gpu::StreamType stream, int n, double* x) -> void;

}  // namespace bluebild
