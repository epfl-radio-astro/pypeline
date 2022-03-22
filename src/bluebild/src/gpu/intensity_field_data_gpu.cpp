#include <complex>
#include <cstddef>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "memory/buffer.hpp"
#include "context_internal.hpp"
#include "gpu/eigensolver_gpu.hpp"
#include "gpu/kernels/min_diff.hpp"
#include "gpu/util/gpu_blas_api.hpp"
#include "gpu/util/gpu_runtime_api.hpp"
#include "gpu/gram_matrix_gpu.hpp"

namespace bluebild {

template <typename T>
auto intensity_field_data_gpu(ContextInternal& ctx, T wl, int m, int n, int nEig,
                              const gpu::ComplexType<T>* s, int lds, const gpu::ComplexType<T>* w,
                              int ldw, const T* xyz, int ldxyz, T* d, gpu::ComplexType<T>* v,
                              int ldv, int nCluster, const T* cluster, int* clusterIndices)
    -> BufferType<int> {
  using ComplexType = gpu::ComplexType<T>;
  using ScalarType = T;


  auto gD = create_buffer<gpu::ComplexType<T>>(ctx.allocators().gpu(), n * n);

  gram_matrix_gpu<T>(ctx, m, n, w, ldw, xyz, ldxyz, wl, gD.get(), n);

  int nEigOut = 0;
  auto devInfo = eigh_gpu<T>(ctx, n, nEig, s, lds, gD.get(), n, &nEigOut, d, v, ldv);

  if (nEigOut) min_diff_1d_gpu(ctx.gpu_stream(), nCluster, cluster, nEigOut, d, clusterIndices);
  return devInfo;
}

template auto intensity_field_data_gpu<float>(ContextInternal& ctx, float wl, int m, int n,
                                              int nEig, const gpu::ComplexType<float>* s, int lds,
                                              const gpu::ComplexType<float>* w, int ldw,
                                              const float* xyz, int ldxyz, float* d,
                                              gpu::ComplexType<float>* v, int ldv, int nCluster,
                                              const float* cluster, int* clusterIndices)
    -> BufferType<int>;

template auto intensity_field_data_gpu<double>(ContextInternal& ctx, double wl, int m, int n,
                                               int nEig, const gpu::ComplexType<double>* s, int lds,
                                               const gpu::ComplexType<double>* w, int ldw,
                                               const double* xyz, int ldxyz, double* d,
                                               gpu::ComplexType<double>* v, int ldv, int nCluster,
                                               const double* cluster, int* clusterIndices)
    -> BufferType<int>;

}  // namespace bluebild
