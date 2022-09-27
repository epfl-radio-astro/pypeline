#pragma once

#include <complex>

#include "bluebild/bluebild.hpp"
#include "bluebild/config.h"
#include "context_internal.hpp"
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild {

template <typename T>
auto virtual_vis_gpu(ContextInternal &ctx, int nFilter,
                     const BluebildFilter *filterHost, int nIntervals,
                     const T *intervalsHost, int ldIntervals, int nEig,
                     const T *D, int nAntenna, const gpu::ComplexType<T> *V,
                     int ldv, int nBeam, const gpu::ComplexType<T> *W, int ldw,
                     gpu::ComplexType<T> *virtVis, int ldVirtVis1,
                     int ldVirtVis2, int ldVirtVis3) -> void;
}  // namespace bluebild
