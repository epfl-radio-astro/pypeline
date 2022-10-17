#include <algorithm>

#include "bluebild/config.h"
#include "gpu/kernels/center_vector.hpp"
#include "gpu/util/gpu_runtime.hpp"
#include "gpu/util/gpu_runtime_api.hpp"

#include <cub/cub.cuh>

namespace bluebild {
namespace gpu {

template <typename T>
static __global__ void
sub_from_vector_kernel(std::size_t n, const T *__restrict__ value,
                       T *__restrict__ vec) {
  const T mean = *value / n;
  for (std::size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
       i += gridDim.x * blockDim.x) {
    vec[i] -= mean;
  }
}

template <typename T>
auto center_vector_get_worksize(gpu::StreamType stream, std::size_t n) -> std::size_t {
  std::size_t size = 0;
  gpu::check_status(::cub::DeviceReduce::Sum<const T *, T *>(
      nullptr, size, nullptr, nullptr, n, stream));
  // Work size includes temporary storage for reduce operation result
  return size + sizeof(T);
}

template <typename T>
auto center_vector(gpu::StreamType stream, std::size_t n, T *vec,
                   std::size_t worksize, void *work) -> void {
  constexpr std::size_t blockSize = 256;
  constexpr std::size_t maxBlocks = 65535;

  worksize -= sizeof(T);

  // To avoid alignment issues for type T, sum up at beginning of work array and
  // provide remaining memory to reduce function
  T *sumPtr = reinterpret_cast<T *>(work);

  gpu::check_status(::cub::DeviceReduce::Sum<const T *, T *>(
      reinterpret_cast<T *>(work) + 1, worksize, vec, sumPtr, n, stream));

  dim3 block(blockSize, 1, 1);

  dim3 grid(std::min<std::size_t>(maxBlocks, (n + block.x - 1) / block.x) / 2 +
                1,
            1, 1);
  gpu::launch_kernel(sub_from_vector_kernel<T>, grid, block, 0, stream, n,
                     sumPtr, vec);
}

template auto center_vector_get_worksize<float>(gpu::StreamType stream,
                                                std::size_t n) -> std::size_t;

template auto center_vector_get_worksize<double>(gpu::StreamType stream,
                                                 std::size_t n) -> std::size_t;

template auto center_vector<float>(gpu::StreamType stream, std::size_t n,
                                   float *vec, std::size_t worksize, void *work)
    -> void;

template auto center_vector<double>(gpu::StreamType stream, std::size_t n,
                                    double *vec, std::size_t worksize,
                                    void *work) -> void;
} // namespace gpu
} // namespace bluebild
