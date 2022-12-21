#pragma once

#include <complex>

#include "bluebild/bluebild.hpp"
#include "bluebild/config.h"
#include "context_internal.hpp"
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild {

template <typename T>
auto virtual_vis_gpu(ContextInternal &ctx, std::size_t nFilter,
                     const BluebildFilter *filterHost, std::size_t nIntervals,
                     const T *intervalsHost, std::size_t ldIntervals,
                     std::size_t nEig, const T *D, std::size_t nAntenna,
                     const gpu::ComplexType<T> *V, std::size_t ldv,
                     std::size_t nBeam, const gpu::ComplexType<T> *W,
                     std::size_t ldw, gpu::ComplexType<T> *virtVis,
                     std::size_t ldVirtVis1, std::size_t ldVirtVis2,
                     std::size_t ldVirtVis3) -> void;
}  // namespace bluebild
