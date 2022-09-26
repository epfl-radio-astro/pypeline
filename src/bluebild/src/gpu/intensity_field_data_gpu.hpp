#pragma once

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "context_internal.hpp"
#include "gpu/util/gpu_runtime_api.hpp"
#include "memory/buffer.hpp"

namespace bluebild {

template <typename T>
auto intensity_field_data_gpu(ContextInternal& ctx, T wl, int m, int n, int nEig,
                              const gpu::ComplexType<T>* s, int lds, const gpu::ComplexType<T>* w,
                              int ldw, const T* xyz, int ldxyz, T* d, gpu::ComplexType<T>* v,
                              int ldv, int nCluster, const T* cluster, int* clusterIndices)
    -> BufferType<int>;

}  // namespace bluebild
