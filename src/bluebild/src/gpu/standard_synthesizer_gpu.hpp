#pragma once

#include <complex>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "context_internal.hpp"
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild {

template <typename T>
auto standard_synthesizer_gpu(ContextInternal& ctx, const T wl, const T* grid,
                              const T* xyz, const T* d, const gpu::ComplexType<T>* v,
                              const gpu::ComplexType<T>* w,
                              const size_t* c_idx, const size_t* c_thick,
                              const size_t Na, const size_t Nb, const size_t Ne,
                              const size_t Nh, const size_t Nl,
                              const size_t Nws, const size_t Nwe, const size_t largest_chunck,
                              T* stats_std_cum, T* stats_lsq_cum) -> void;
}  // namespace bluebild
