#include <complex>
#include <cstddef>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "memory/buffer.hpp"
#include "context_internal.hpp"
#include "gpu/kernels/standard_synthesizer.hpp"
#include "gpu/util/gpu_blas_api.hpp"
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
                              T* stats_std_cum, T* stats_lsq_cum) -> void {

  using ComplexType = gpu::ComplexType<T>;

  // Nw is now the split thickness
  const size_t Nw = Nwe - Nws;

  // Allocate for intermediate matrices
  auto pD  = create_buffer<ComplexType>(ctx.allocators().gpu(), Na * Nh * largest_chunck);
  auto pwD = create_buffer<ComplexType>(ctx.allocators().gpu(), Nb * Nh * largest_chunck);
  auto eD  = create_buffer<ComplexType>(ctx.allocators().gpu(), Ne * Nh * largest_chunck);

  standard_synthesizer_p_gpu(ctx.gpu_stream(), wl, grid, xyz, Na, Nh, Nw, pD.get());

  auto hd_ws  = create_buffer<const gpu::ComplexType<T>*>(ctx.allocators().host(), largest_chunck);
  auto hd_ps  = create_buffer<const gpu::ComplexType<T>*>(ctx.allocators().host(), largest_chunck);
  auto hd_pws = create_buffer<const gpu::ComplexType<T>*>(ctx.allocators().host(), largest_chunck);
  auto hd_vs  = create_buffer<const gpu::ComplexType<T>*>(ctx.allocators().host(), largest_chunck);
  auto hd_es  = create_buffer<const gpu::ComplexType<T>*>(ctx.allocators().host(), largest_chunck);

  size_t idx_ip, idx_ipw, idx_ie;
  for (uint k=0; k<Nw; k++) {
      hd_ws[k]  = w; // invariant
      hd_vs[k]  = v; // invariant
      idx_ip    = k * Na * Nh;
      idx_ipw   = k * Nh * Nb;
      idx_ie    = k * Nh * Ne;
      hd_ps[k]  = &pD[idx_ip];
      hd_pws[k] = &pwD[idx_ipw];
      hd_es[k]  = &eD[idx_ie];
  }

  auto d_ws  = create_buffer<const gpu::ComplexType<T>*>(ctx.allocators().gpu(), largest_chunck);
  auto d_ps  = create_buffer<const gpu::ComplexType<T>*>(ctx.allocators().gpu(), largest_chunck);
  auto d_pws = create_buffer<gpu::ComplexType<T>*>(ctx.allocators().gpu(), largest_chunck);
  auto d_vs  = create_buffer<const gpu::ComplexType<T>*>(ctx.allocators().gpu(), largest_chunck);
  auto d_es  = create_buffer<gpu::ComplexType<T>*>(ctx.allocators().gpu(), largest_chunck);

  gpu::check_status(gpu::memcpy_async(d_ws.get(), hd_ws.get(), Nw * sizeof(gpu::ComplexType<T>*),
                                      gpu::flag::MemcpyHostToDevice, ctx.gpu_stream()));
  gpu::check_status(gpu::memcpy_async(d_ps.get(), hd_ps.get(), Nw * sizeof(gpu::ComplexType<T>*),
                                      gpu::flag::MemcpyHostToDevice, ctx.gpu_stream()));
  gpu::check_status(gpu::memcpy_async(d_pws.get(), hd_pws.get(), Nw * sizeof(gpu::ComplexType<T>*),
                                      gpu::flag::MemcpyHostToDevice, ctx.gpu_stream()));
  gpu::check_status(gpu::memcpy_async(d_vs.get(), hd_vs.get(), Nw * sizeof(gpu::ComplexType<T>*),
                                      gpu::flag::MemcpyHostToDevice, ctx.gpu_stream()));
  gpu::check_status(gpu::memcpy_async(d_es.get(), hd_es.get(), Nw * sizeof(gpu::ComplexType<T>*),
                                      gpu::flag::MemcpyHostToDevice, ctx.gpu_stream()));
  
  const ComplexType alpha{1.0, 0.0};
  const ComplexType beta{0.0, 0.0};

  gpu::blas::check_status(gpu::blas::gemm_batched(ctx.gpu_blas_handle(),
                                                  gpu::blas::operation::Transpose,
                                                  gpu::blas::operation::None,
                                                  (int)Nb, (int)Nh, (int)Na,
                                                  &alpha,
                                                  d_ws.get(), (int)Na,
                                                  d_ps.get(), (int)Na,
                                                  &beta,
                                                  d_pws.get(), (int)Nb,
                                                  (int)Nw));

  gpu::blas::check_status(gpu::blas::gemm_batched(ctx.gpu_blas_handle(),
                                                  gpu::blas::operation::Transpose,
                                                  gpu::blas::operation::None,
                                                  (int)Ne, (int)Nh, (int)Nb,
                                                  &alpha,
                                                  d_vs.get(), (int)Nb,
                                                  d_pws.get(), (int)Nb,
                                                  &beta,
                                                  d_es.get(), (int)Ne,
                                                  (int)Nw));

  // Compute stats (stacking Ne levels down to Nl according to c_idx)
  standard_synthesizer_stats_gpu(ctx.gpu_stream(), d, eD.get(), Ne, Nh, Nl, Nw, c_idx, c_thick,
                                 stats_std_cum, stats_lsq_cum);
}

template auto standard_synthesizer_gpu<float>(ContextInternal& ctx, const float wl, const float* grid,
                                              const float* xyz, const float* d, const gpu::ComplexType<float>* v, 
                                              const gpu::ComplexType<float>* w,
                                              const size_t* c_idx, const size_t* c_thick,
                                              const size_t Na, const size_t Nb, const size_t Ne,
                                              const size_t Nh, const size_t Nl,
                                              const size_t Nws, const size_t Nwe, const size_t largest_chunck,
                                              float* stats_std_cum, float* stats_lsq_cum) -> void;

template auto standard_synthesizer_gpu<double>(ContextInternal& ctx, const double wl, const double* grid,
                                               const double* xyz, const double* d, const gpu::ComplexType<double>* v, 
                                               const gpu::ComplexType<double>* w,
                                               const size_t* c_idx, const size_t* c_thick,
                                               const size_t Na, const size_t Nb, const size_t Ne,
                                               const size_t Nh, const size_t Nl,
                                               const size_t Nws, const size_t Nwe, const size_t largest_chunk,
                                               double* stats_std_cum, double* stats_lsq_cum) -> void;
}  // namespace bluebild
