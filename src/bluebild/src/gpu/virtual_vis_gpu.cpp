#include <complex>

#include "bluebild/config.h"
#include "context_internal.hpp"
#include "gpu/kernels/scale_matrix.hpp"
#include "gpu/kernels/apply_filter.hpp"
#include "gpu/util/gpu_blas_api.hpp"
#include "gpu/util/gpu_runtime_api.hpp"
#include "gpu/virtual_vis_gpu.hpp"
#include "host/virtual_vis_host.hpp"
#include "memory/buffer.hpp"

namespace bluebild {

template <typename T>
auto virtual_vis_gpu(ContextInternal &ctx, int nFilter,
                     const BluebildFilter *filterHost, int nIntervals,
                     const T *intervalsHost, int ldIntervals, int nEig,
                     const T *D, int nAntenna, const gpu::ComplexType<T> *V,
                     int ldv, int nBeam, const gpu::ComplexType<T> *W, int ldw,
                     gpu::ComplexType<T> *virtVis, int ldVirtVis1,
                     int ldVirtVis2, int ldVirtVis3) -> void {
  using ComplexType = gpu::ComplexType<T>;

  const auto zero = ComplexType{0, 0};
  const auto one = ComplexType{1, 0};

  BufferType<gpu::ComplexType<T>> VUnbeamBuffer;
  if (W) {
    VUnbeamBuffer =
        create_buffer<gpu::ComplexType<T>>(ctx.allocators().gpu(), nAntenna * nEig);

    gpu::blas::check_status(
        gpu::blas::gemm(ctx.gpu_blas_handle(), gpu::blas::operation::None,
                        gpu::blas::operation::None, nAntenna, nEig, nBeam, &one,
                        W, ldw, V, ldv, &zero, VUnbeamBuffer.get(), nAntenna));
    V = VUnbeamBuffer.get();
    ldv = nAntenna;
  }
  // V is alwayts of shape (nAntenna, nEig) from here on

  auto VMulDBuffer = create_buffer<gpu::ComplexType<T>>(ctx.allocators().gpu(),
                                                    nEig * nAntenna);

  auto DBufferHost = create_buffer<T>(ctx.allocators().pinned(), nEig);
  auto DFilteredBuffer = create_buffer<T>(ctx.allocators().gpu(), nEig);

  gpu::check_status(gpu::memcpy_async(DBufferHost.get(), D, nEig * sizeof(T),
                                      gpu::flag::MemcpyDeviceToHost,
                                      ctx.gpu_stream()));
  // Make sure D is available on host
  gpu::check_status(gpu::stream_synchronize(ctx.gpu_stream()));

  for (std::size_t i = 0; i < static_cast<std::size_t>(nFilter); ++i) {
    gpu::apply_filter(ctx.gpu_stream(), filterHost[i], nEig, D,
                      DFilteredBuffer.get());

    for (std::size_t j = 0; j < static_cast<std::size_t>(nIntervals); ++j) {
      std::size_t start, size;
      std::tie(start, size) = find_interval_indices(
          nEig, DBufferHost.get(),
          intervalsHost[j * static_cast<std::size_t>(ldIntervals)],
          intervalsHost[j * static_cast<std::size_t>(ldIntervals) + 1]);

      auto virtVisCurrent = virtVis + i * static_cast<std::size_t>(ldVirtVis1) +
                            j * static_cast<std::size_t>(ldVirtVis2);
      if (size) {
        // Multiply each col of V with the selected eigenvalue
        gpu::scale_matrix<T>(ctx.gpu_stream(), nAntenna, size, V + start * ldv,
                          ldv, DFilteredBuffer.get() + start, VMulDBuffer.get(),
                          nAntenna);

        // Matrix multiplication of the previously scaled V and the original V
        // with the selected eigenvalues
        gpu::blas::check_status(gpu::blas::gemm(
            ctx.gpu_blas_handle(), gpu::blas::operation::None,
            gpu::blas::operation::ConjugateTranspose, nAntenna, nAntenna, size,
            &one, VMulDBuffer.get(), nAntenna, V + start * ldv, ldv, &zero,
            virtVisCurrent, ldVirtVis3));

      } else {
        gpu::check_status(gpu::memset_2d_async(
            virtVisCurrent, ldVirtVis3 * sizeof(ComplexType), 0,
            nAntenna * sizeof(ComplexType), nAntenna));
      }
    }
  }
}

template auto virtual_vis_gpu<float>(
    ContextInternal &ctx, int nFilter, const BluebildFilter *filter,
    int nIntervals, const float *intervals, int ldIntervals, int nEig,
    const float *D, int nAntenna, const gpu::ComplexType<float> *V, int ldv,
    int nBeam, const gpu::ComplexType<float> *W, int ldw,
    gpu::ComplexType<float> *virtVis, int ldVirtVis1, int ldVirtVis2,
    int ldVirtVis3) -> void;

template auto virtual_vis_gpu<double>(ContextInternal &ctx, int nFilter,
                              const BluebildFilter *filter, int nIntervals,
                              const double *intervals, int ldIntervals, int nEig,
                              const double *D, int nAntenna,
                              const gpu::ComplexType<double> *V, int ldv,
                              int nBeam, const gpu::ComplexType<double> *W,
                              int ldw, gpu::ComplexType<double> *virtVis,
                              int ldVirtVis1, int ldVirtVis2, int ldVirtVis3)
    -> void;

}  // namespace bluebild
