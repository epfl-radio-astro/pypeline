#pragma once

#include "bluebild/config.h"
#include "context_internal.hpp"
#include "memory/buffer.hpp"
#include "gpu/util/gpu_runtime_api.hpp"


namespace bluebild {

template <typename T> class NufftSynthesisGPU {
public:
  NufftSynthesisGPU(std::shared_ptr<ContextInternal> ctx, T tol,
                    std::size_t nAntenna, std::size_t nBeam,
                    std::size_t nIntervals, std::size_t nFilter,
                    const BluebildFilter *filterHost, std::size_t nPixel,
                    const T *lmnX, const T *lmnY, const T *lmnZ);

  auto collect(std::size_t nEig, T wl, const T *intervals,
               std::size_t ldIntervals, const gpu::ComplexType<T> *s,
               std::size_t lds, const gpu::ComplexType<T> *w, std::size_t ldw,
               const T *xyz, std::size_t ldxyz, const T *uvw, std::size_t lduvw)
      -> void;

  auto get(BluebildFilter f, T *outHostOrDevice, std::size_t ld) -> void;

  auto context() -> ContextInternal & { return *ctx_; }

private:
  auto computeNufft() -> void;

  std::shared_ptr<ContextInternal> ctx_;
  const T tol_;
  const std::size_t nIntervals_, nFilter_, nPixel_, nAntenna_, nBeam_;
  BufferType<BluebildFilter> filterHost_;
  BufferType<T> lmnX_, lmnY_, lmnZ_;

  std::size_t nMaxInputCount_, inputCount_;
  BufferType<gpu::ComplexType<T>> virtualVis_;
  BufferType<T> uvwX_, uvwY_, uvwZ_;
  BufferType<T> img_;

};

} // namespace bluebild
