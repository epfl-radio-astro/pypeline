#include <cmath>
#include <complex>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <fstream>
#include <iostream>

#include "bluebild/nufft_3d3.hpp"
#include "bluebild/context.hpp"
#include "gtest/gtest.h"


namespace {
  template<typename T>
  struct Nufft3d3Type {
    using type = bluebild::Nufft3d3;
  };
  template<>
  struct Nufft3d3Type<float> {
    using type = bluebild::Nufft3d3f;
  };
}

template <typename T>
class Nufft3d3Test : public ::testing::TestWithParam<std::tuple<BluebildProcessingUnit, int, int>> {
protected:
  using ValueType = T;

  Nufft3d3Test()
      : ctx_(std::get<0>(GetParam())), M_(std::get<1>(GetParam())),
        N_(std::get<2>(GetParam())) {}

  void run() {
    const T tol = 0.001;
    const int iflag = 1;
    std::mt19937 randGen(M_ + N_);
    std::uniform_real_distribution<T> uniDist(0, 1);

    std::vector<T> x(M_), y(M_), z(M_);
    std::vector<T> s(N_), t(N_), u(N_);

    // const T inScale = M_ * 0.1 + 3;
    const T inScale = 1;
    for(int i=0; i < M_; ++i) {
      x[i] = inScale * uniDist(randGen) + inScale;
      y[i] = inScale * uniDist(randGen) + inScale;
      z[i] = inScale * uniDist(randGen) + inScale;
    }

    // const T outScale = N_ * 0.1 + 3;
    const T outScale = 1;
    for(int i=0; i < N_; ++i) {
      s[i] = outScale * uniDist(randGen) + outScale;
      t[i] = outScale * uniDist(randGen) + outScale;
      u[i] = outScale * uniDist(randGen) + outScale;
    }


    std::vector<std::complex<T>> cj(M_), fk(N_);
    for(int i=0; i < M_; ++i) {
      cj[i] = 10 * uniDist(randGen);
    }


    // create and execute plan
    typename Nufft3d3Type<T>::type plan(ctx_, iflag, tol, 1, M_, x.data(), y.data(), z.data(), N_, s.data(), t.data(), u.data());
    plan.execute(cj.data(), fk.data());

    // compute inf norm of output for relative comparison
    T infNorm = 0;
    for(int i=0; i < N_; ++i) {
      infNorm = std::max<T>(infNorm, std::abs(fk[i]));
    }

    for (int i = 0; i < N_; ++i) {
      std::complex<T> ref(0);
      for (int j = 0; j < M_; ++j)
        ref +=
            cj[j] *
            std::exp(std::complex<T>(0, iflag * (x[j] * s[i] + y[j] * t[i] +
                                                 z[j] * u[i]))); // crude direct
      ASSERT_LT(std::abs(fk[i] - ref) / infNorm, tol);
    }
  }

  bluebild::Context ctx_;
  int M_, N_;
};

using NufftDouble = Nufft3d3Test<double>;
using NufftSingle = Nufft3d3Test<float>;

TEST_P(NufftDouble, Nufft3d3) { this->run(); }
TEST_P(NufftSingle, Nufft3d3) { this->run(); }

static auto param_type_names(
    const ::testing::TestParamInfo<std::tuple<BluebildProcessingUnit, int, int>>
        &info) -> std::string {
  std::stringstream stream;

  if (std::get<0>(info.param) == BLUEBILD_PU_CPU)
    stream << "CPU_";
  else
    stream << "GPU_";
  stream << "m_" << std::get<1>(info.param);
  stream << "_n_" << std::get<2>(info.param);

  return stream.str();
}

#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
#define TEST_PROCESSING_UNITS BLUEBILD_PU_CPU, BLUEBILD_PU_GPU
#else
#define TEST_PROCESSING_UNITS BLUEBILD_PU_CPU
#endif

INSTANTIATE_TEST_SUITE_P(
    Nufft, NufftSingle,
    ::testing::Combine(::testing::Values(TEST_PROCESSING_UNITS),
                       ::testing::Values(100, 200), ::testing::Values(50, 256)),
    param_type_names);

INSTANTIATE_TEST_SUITE_P(
    Nufft, NufftDouble,
    ::testing::Combine(::testing::Values(TEST_PROCESSING_UNITS),
                       ::testing::Values(100, 200), ::testing::Values(50, 256)),
    param_type_names);
