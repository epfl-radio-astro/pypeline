#pragma once

#include "gpu/util/gpu_runtime_api.hpp"
#include "bluebild/config.h"

#ifdef BLUEBILD_ROCM
#include <hip/hip_runtime.h>
#endif

namespace bluebild {
namespace gpu {

#ifdef BLUEBILD_CUDA
template <typename F, typename... ARGS>
inline auto launch_kernel(F func, const dim3 threadGrid, const dim3 threadBlock,
                          const size_t sharedMemoryBytes, const gpu::StreamType stream,
                          ARGS&&... args) -> void {
#ifndef NDEBUG
  gpu::device_synchronize();
  gpu::check_status(gpu::get_last_error());  // before
#endif
  func<<<threadGrid, threadBlock, sharedMemoryBytes, stream>>>(std::forward<ARGS>(args)...);
#ifndef NDEBUG
  gpu::device_synchronize();
  gpu::check_status(gpu::get_last_error());  // after
#endif
}
#endif

#ifdef BLUEBILD_ROCM
template <typename F, typename... ARGS>
inline auto launch_kernel(F func, const dim3 threadGrid, const dim3 threadBlock,
                          const size_t sharedMemoryBytes, const gpu::StreamType stream,
                          ARGS&&... args) -> void {
#ifndef NDEBUG
  gpu::device_synchronize();
  gpu::check_status(gpu::get_last_error());  // before
#endif
  hipLaunchKernelGGL(func, threadGrid, threadBlock, sharedMemoryBytes, stream,
                     std::forward<ARGS>(args)...);
#ifndef NDEBUG
  gpu::device_synchronize();
  gpu::check_status(gpu::get_last_error());  // after
#endif
}
#endif

}  // namespace gpu
}  // namespace bluebild

