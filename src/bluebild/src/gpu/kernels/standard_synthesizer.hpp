#pragma once

#include <complex>
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild{

template <typename T>
auto standard_synthesizer_p_gpu(gpu::StreamType stream, const T wl, const T* grid,
                                const T* xyz, const size_t Na, const size_t Nh, const size_t Nw,
                                gpu::ComplexType<T>* p) -> void;

template <typename T>
auto standard_synthesizer_stats_gpu(gpu::StreamType stream, const T* d, const gpu::ComplexType<T>* e,
                                    const size_t Ne, const size_t Nh, const size_t Nl, const size_t Nw,
                                    const size_t* c_idx, const size_t* c_thickness,
                                    T* stats_std_cum, T* stats_lsq_cum) -> void;

}
