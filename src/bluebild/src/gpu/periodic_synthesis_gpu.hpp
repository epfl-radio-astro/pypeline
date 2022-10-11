#pragma once

#include "bluebild/config.h"
#include "context_internal.hpp"
#include "memory/buffer.hpp"
#include "gpu/util/gpu_runtime_api.hpp"


namespace bluebild {

template <typename T> class PeriodicSynthesisGPU {
public:
  PeriodicSynthesisGPU(std::shared_ptr<ContextInternal> ctx, T tol,
                        int nAntenna, int nBeam,
                        int nIntervals, int nFilter,
                        const BluebildFilter *filterHost, int nPixel,
                        const T *lmnX, const T *lmnY, const T *lmnZ);

  auto collect(int nEig, T wl, const T *intervals,
               int ldIntervals, const gpu::ComplexType<T> *s,
               int lds, const gpu::ComplexType<T> *w, int ldw,
               const T *xyz, int ldxyz, const T *uvwX, const T *uvwY,
               const T *uvwZ) -> void;

  auto get(BluebildFilter f, T *outHostOrDevice, int ld) -> void;

  auto context() -> ContextInternal & { return *ctx_; }

private:
  auto computeNufft() -> void;

  std::shared_ptr<ContextInternal> ctx_;
  const T tol_;
  const int nIntervals_, nFilter_, nPixel_, nAntenna_, nBeam_;
  BufferType<BluebildFilter> filterHost_;
  BufferType<T> lmnX_, lmnY_, lmnZ_;

  int nMaxInputCount_, inputCount_;
  BufferType<gpu::ComplexType<T>> virtualVis_;
  BufferType<T> uvwX_, uvwY_, uvwZ_;
  BufferType<T> img_;

};

} // namespace bluebild
