#pragma once

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "context_internal.hpp"
#include "gpu/util/gpu_runtime_api.hpp"
#include "memory/buffer.hpp"

namespace bluebild {

template <typename T>
auto sensitivity_field_data_gpu(ContextInternal &ctx, T wl, std::size_t m,
                                std::size_t n, std::size_t nEig,
                                const gpu::ComplexType<T> *w, std::size_t ldw,
                                const T *xyz, std::size_t ldxyz, T *d,
                                gpu::ComplexType<T> *v, std::size_t ldv)
    -> void;
}
