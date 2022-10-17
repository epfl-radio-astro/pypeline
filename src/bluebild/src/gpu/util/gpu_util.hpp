#pragma once

#include <tuple>

#include "bluebild/config.h"
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild {
template <typename T>
auto is_device_ptr(const T* ptr) -> bool {
  gpu::PointerAttributes attr;
  auto status = gpu::pointer_get_attributes(&attr, static_cast<const void*>(ptr));

  if (status != gpu::status::Success) {
    // throw error if unexpected error
    if (status != gpu::status::ErrorInvalidValue) gpu::check_status(status);
    // clear error from cache and return otherwise
    std::ignore = gpu::get_last_error();
    return false;
  }

  // get memory type - cuda 10 changed attribute name
#if defined(BLUEBILD_CUDA) && (CUDART_VERSION >= 10000)
  auto memoryType = attr.type;
#else
  auto memoryType = attr.memoryType;
#endif

  if (memoryType == gpu::flag::MemoryTypeDevice) {
    return true;
  }
  return false;
}
}  // namespace bluebild
