#pragma once

#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild {
namespace gpu {
template <typename T>
auto add_vector(gpu::StreamType stream, int n, const gpu::ComplexType<T> *a,
                T *b) -> void;
}
}
