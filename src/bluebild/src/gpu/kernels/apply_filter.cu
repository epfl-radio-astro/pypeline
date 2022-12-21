#include <algorithm>

#include "bluebild//config.h"
#include "gpu/kernels/apply_filter.hpp"
#include "gpu/util/gpu_runtime.hpp"
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild {
namespace gpu {

static __device__ auto calc_sqrt(float x) -> float { return sqrtf(x); }

static __device__ auto calc_sqrt(double x) -> double { return sqrt(x); }

template <typename T>
__global__ void apply_filter_std_kernel(std::size_t n, T *out) {
  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
       i += gridDim.x * blockDim.x) {
    out[i] = 1;
  }
}

template <typename T>
__global__ void apply_filter_sqrt_kernel(std::size_t n,
                                         const T *__restrict__ in, T *out) {
  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
       i += gridDim.x * blockDim.x) {
    out[i] = calc_sqrt(in[i]);
  }
}

template <typename T>
__global__ void apply_filter_inv_kernel(std::size_t n, const T *__restrict__ in,
                                        T *out) {
  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
       i += gridDim.x * blockDim.x) {
    const auto value = in[i];
    if (value)
      out[i] = static_cast<T>(1) / value;
    else
      out[i] = 0;
  }
}

template <typename T>
__global__ void apply_filter_inv_sq_kernel(std::size_t n,
                                           const T *__restrict__ in, T *out) {
  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
       i += gridDim.x * blockDim.x) {
    const auto value = in[i];
    if (value)
      out[i] = static_cast<T>(1) / (value * value);
    else
      out[i] = 0;
  }
}

template <typename T>
auto apply_filter(gpu::StreamType stream, BluebildFilter filter, std::size_t n,
                  const T *in, T *out) -> void {
  constexpr std::size_t blockSize = 256;
  constexpr std::size_t maxBlocks = 65535;

  dim3 block(blockSize, 1, 1);
  dim3 grid(std::min<std::size_t>(maxBlocks, (n + block.x - 1) / block.x), 1,
            1);

  switch (filter) {
  case BLUEBILD_FILTER_STD: {
    gpu::launch_kernel(apply_filter_std_kernel<T>, grid, block, 0, stream, n,
                       out);
    break;
  }
  case BLUEBILD_FILTER_SQRT: {
    gpu::launch_kernel(apply_filter_sqrt_kernel<T>, grid, block, 0, stream, n,
                       in, out);
    break;
  }
  case BLUEBILD_FILTER_INV: {
    gpu::launch_kernel(apply_filter_inv_kernel<T>, grid, block, 0, stream, n,
                       in, out);
    break;
  }
  case BLUEBILD_FILTER_INV_SQ: {
    gpu::launch_kernel(apply_filter_inv_sq_kernel<T>, grid, block, 0, stream, n,
                       in, out);
    break;
  }
  case BLUEBILD_FILTER_LSQ: {
    gpu::check_status(gpu::memcpy_async(
        out, in, n * sizeof(T), gpu::flag::MemcpyDeviceToDevice, stream));
    break;
  }
  }
}

template auto apply_filter<float>(gpu::StreamType stream, BluebildFilter filter,
                                  std::size_t n, const float *in, float *out)
    -> void;

template auto apply_filter<double>(gpu::StreamType stream,
                                   BluebildFilter filter, std::size_t n,
                                   const double *in, double *out) -> void;

} // namespace gpu
} // namespace bluebild
