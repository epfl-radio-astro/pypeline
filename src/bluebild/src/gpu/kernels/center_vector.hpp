#pragma once

#include <cstddef>
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild{
namespace gpu{
template <typename T>
auto center_vector_get_worksize(gpu::StreamType stream, std::size_t n)
    -> std::size_t;

template <typename T>
auto center_vector(gpu::StreamType stream, std::size_t n, T *vec,
                   std::size_t worksize, void *work) -> void;
} // namespace gpu
} // namespace bluebild
