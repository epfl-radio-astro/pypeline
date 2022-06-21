#pragma once

#include <complex>
#include <memory>
#include <functional>

#include "bluebild/config.h"
#include "bluebild/exceptions.hpp"


namespace bluebild {

template <typename T> class Nufft3d3Host {
public:
  using planType = std::unique_ptr<void, std::function<void(void *)>>;

  Nufft3d3Host(int iflag, T tol, int numTrans, int M, const T *x,
               const T *y, const T *z, int N, const T *s,
               const T *t, const T *u);

  void execute(const std::complex<T> *cj, std::complex<T> *fk);


private:
  planType plan_;
};

} // namespace bluebild
