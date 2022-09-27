#pragma once

#include "bluebild/bluebild.hpp"
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild {
namespace gpu {
template <typename T>
auto apply_filter(gpu::StreamType stream, BluebildFilter filter, int n,
                  const T *in, T *out) -> void;
}
}
