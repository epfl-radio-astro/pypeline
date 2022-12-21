#pragma once

#include <complex>
#include <cstddef>
#include <functional>
#include <memory>

#include "bluebild/config.h"
#include "bluebild/context.hpp"

namespace bluebild {

template <typename T> class BLUEBILD_EXPORT NufftSynthesis {
public:
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);
  using valueType = T;

  NufftSynthesis(Context &ctx, T tol, std::size_t nAntenna, std::size_t nBeam,
                 std::size_t nIntervals, std::size_t nFilter,
                 const BluebildFilter *filter, std::size_t nPixel,
                 const T *lmnX, const T *lmnY, const T *lmnZ);

  auto collect(std::size_t nEig, T wl, const T *intervals,
               std::size_t ldIntervals, const std::complex<T> *s,
               std::size_t lds, const std::complex<T> *w, std::size_t ldw,
               const T *xyz, std::size_t ldxyz, const T *uvw, std::size_t lduvw)
      -> void;

  auto get(BluebildFilter f, T *out, std::size_t ld) -> void;

private:
  std::unique_ptr<void, std::function<void(void *)>> plan_;
};

} // namespace bluebild
