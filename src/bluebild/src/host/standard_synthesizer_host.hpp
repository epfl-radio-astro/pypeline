#pragma once

#include <complex>
#include <cstddef>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "context_internal.hpp"

namespace bluebild {

template <typename T>
auto mean_center(T* __restrict__ xyz, const T* __restrict__ xyz_, const size_t N) -> void;


template <typename T>
auto standard_synthesizer_host(ContextInternal& ctx,
                               const T* __restrict__ d,
                               const std::complex<T>* __restrict__ v,
                               const T* __restrict__ xyz,
                               const std::complex<T>* __restrict__ w,
                               const std::size_t* __restrict__ c_idx,
                               const std::size_t Nl,
                               const T* __restrict__ grid,
                               const T wl,
                               const std::size_t Na,
                               const std::size_t Nb,
                               const std::size_t Nc,
                               const std::size_t Ne,
                               const std::size_t Nh,
                               const std::size_t Nw,
                               T* __restrict__ stats_std_cum,
                               T* __restrict__ stats_lsq_cum) -> void;

}  // namespace bluebild
