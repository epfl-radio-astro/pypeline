#pragma once

#include <cstddef>

#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild {
namespace gpu {
template <typename T>
auto add_vector_real(gpu::StreamType stream, std::size_t n,
                     const gpu::ComplexType<T> *a, T *b) -> void;
}
}
