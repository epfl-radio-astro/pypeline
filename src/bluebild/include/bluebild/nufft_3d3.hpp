#pragma once

#include <complex>
#include <functional>
#include <memory>

#include "bluebild/config.h"
#include "bluebild/context.hpp"

namespace bluebild {

class BLUEBILD_EXPORT Nufft3d3 {
public:
  /**
   * Create plan for a type 3 nufft transform in 3D in single precision.
   *
   * @param[in] ctx Context handle.
   * @param[in] iflag Sign in exponential. Either +1 or -1.
   * @param[in] tol Target preceision tolorance.
   * @param[in] numTrans Number of transforms to compute together.
   * @param[in] M Number of input points.
   * @param[in] x Input coordinates in x.
   * @param[in] y Input coordinates in y.
   * @param[in] z Input coordinates in z.
   * @param[in] N Number of output points.
   * @param[in] s Input coordinates in s.
   * @param[in] t Input coordinates in t.
   * @param[in] u Input coordinates in u.
   */
  Nufft3d3(const Context &ctx, int iflag, double tol, int numTrans, int M,
           const double *x, const double *y, const double *z, int N,
           const double *s, const double *t, const double *u);

  /**
   * Execute a nufft3d3 plan in double precision.
   *
   * @param[in] plan Plan handle.
   * @param[in] cj Input data.
   */
  void execute(const std::complex<double> *cj, std::complex<double> *fk);

private:
  std::unique_ptr<void, std::function<void(void *)>> plan_;
};

class BLUEBILD_EXPORT Nufft3d3f {
public:
  /**
   * Create plan for a type 3 nufft transform in 3D in single precision.
   *
   * @param[in] ctx Context handle.
   * @param[in] iflag Sign in exponential. Either +1 or -1.
   * @param[in] tol Target preceision tolorance.
   * @param[in] numTrans Number of transforms to compute together.
   * @param[in] M Number of input points.
   * @param[in] x Input coordinates in x.
   * @param[in] y Input coordinates in y.
   * @param[in] z Input coordinates in z.
   * @param[in] N Number of output points.
   * @param[in] s Input coordinates in s.
   * @param[in] t Input coordinates in t.
   * @param[in] u Input coordinates in u.
   */
  Nufft3d3f(const Context &ctx, int iflag, float tol, int numTrans, int M,
            const float *x, const float *y, const float *z, int N,
            const float *s, const float *t, const float *u);

  /**
   * Execute a nufft3d3 plan in double precision.
   *
   * @param[in] plan Plan handle.
   * @param[in] cj Input data.
   */
  void execute(const std::complex<float> *cj, std::complex<float> *fk);

private:
  std::unique_ptr<void, std::function<void(void *)>> plan_;
};

} // namespace bluebild
