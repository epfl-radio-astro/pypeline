#include <cmath>
#include <complex>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <fstream>
#include <iostream>

#include "bluebild/bluebild.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"



static auto get_lofar_json(int nStation) -> const nlohmann::json& {
  static nlohmann::json data = []() {
    std::ifstream file(std::string(BLUEBILD_TEST_DATA_DIR) + "/data_lofar.json");
    nlohmann::json j;
    file >> j;
    return j;
  }();

  for(const auto& it : data) {
    if(it["n"] == nStation) return it["data"];
  }
  throw std::runtime_error("Failed to find station data in json");
  return data;
}

template <typename T, typename JSON>
static auto read_json_complex_2d(JSON jReal, JSON jImag) -> std::vector<std::complex<T>> {
  std::vector<std::complex<T>> w;

  auto wRealCol = jReal.begin();
  auto wImagCol = jImag.begin();
  for (; wRealCol != jReal.end(); ++wRealCol, ++wImagCol) {
    auto wReal = wRealCol->begin();
    auto wImag = wImagCol->begin();
    for (; wReal != wRealCol->end(); ++wReal, ++wImag) {
      w.emplace_back(*wReal, *wImag);
    }
  }
  return w;
}

template <typename T, typename JSON>
static auto read_json_scalar_2d(JSON j) -> std::vector<T> {
  std::vector<T> w;

  for (auto &col : j) {
    for (auto &val : col) {
      w.emplace_back(val);
    }
  }

  return w;
}

template <typename T>
class LofarTest : public ::testing::TestWithParam<std::tuple<int, BluebildProcessingUnit>> {
protected:
  using ValueType = T;

  LofarTest() : data_(get_lofar_json(std::get<0>(GetParam()))) {
    if (bluebild_ctx_create(std::get<1>(GetParam()), &ctx_) != BLUEBILD_SUCCESS)
      throw std::runtime_error("Failed to create context.");
  }

  auto gram_matrix() -> void {
    auto wl = ValueType(data_["sensitivity"]["wl"]);

    for (const auto& sensData : data_["sensitivity"]["data"]) {
      auto m = sensData["XYZ"][0].size();
      auto n = sensData["G_real"].size();
      auto xyz = read_json_scalar_2d<ValueType>(sensData["XYZ"]);
      auto w = read_json_complex_2d<ValueType>(sensData["W_real"], sensData["W_imag"]);
      auto gRef = read_json_complex_2d<ValueType>(sensData["G_real"], sensData["G_imag"]);

      std::vector<std::complex<ValueType>> g(n * n);

      if (std::is_same<ValueType, float>::value)
        ASSERT_EQ(bluebild_gram_matrix_s(ctx_, m, n, w.data(), m,
                                         reinterpret_cast<float*>(xyz.data()), m, wl, g.data(), n),
                  BLUEBILD_SUCCESS);
      else
        ASSERT_EQ(bluebild_gram_matrix_d(ctx_, m, n, w.data(), m,
                                         reinterpret_cast<double*>(xyz.data()), m, wl, g.data(), n),
                  BLUEBILD_SUCCESS);

      for (std::size_t i = 0; i < g.size(); ++i) {
        ASSERT_NEAR(g[i].real(), gRef[i].real(), 10);
        ASSERT_NEAR(g[i].imag(), gRef[i].imag(), 10);
      }
    }
  }

  auto sensitivity_field_data() -> void {
    auto wl = ValueType(data_["sensitivity"]["wl"]);
    auto nEig = int(data_["sensitivity"]["n_eig"]);

    for (const auto& sensData : data_["sensitivity"]["data"]) {
      auto m = sensData["XYZ"][0].size();
      auto n = sensData["G_real"].size();
      auto xyz = read_json_scalar_2d<ValueType>(sensData["XYZ"]);
      auto w = read_json_complex_2d<ValueType>(sensData["W_real"], sensData["W_imag"]);
      auto vRef = read_json_complex_2d<ValueType>(sensData["V_real"], sensData["V_imag"]);
      auto dRef = std::vector<ValueType>(sensData["D"].begin(), sensData["D"].end());

      std::vector<std::complex<ValueType>> v(nEig * n);
      std::vector<ValueType> d(nEig);

      if (std::is_same<ValueType, float>::value)
        ASSERT_EQ(bluebild_sensitivity_field_data_s(
                      ctx_, wl, m, n, nEig, w.data(), m, reinterpret_cast<float*>(xyz.data()), m,
                      reinterpret_cast<float*>(d.data()), v.data(), n),
                  BLUEBILD_SUCCESS);
      else
        ASSERT_EQ(bluebild_sensitivity_field_data_d(
                      ctx_, wl, m, n, nEig, w.data(), m, reinterpret_cast<double*>(xyz.data()), m,
                      reinterpret_cast<double*>(d.data()), v.data(), n),
                  BLUEBILD_SUCCESS);

      for (std::size_t i = 0; i < d.size(); ++i) {
        ASSERT_NEAR(d[i], dRef[i], 0.1);
      }

      for (std::size_t i = 0; i < v.size(); ++i) {
        // Eigenvectors may be mirrored
        ASSERT_NEAR(std::abs(v[i].real()), std::abs(vRef[i].real()), 1.0);
        ASSERT_NEAR(std::abs(v[i].imag()), std::abs(vRef[i].imag()), 1.0);
      }
    }
  }

  auto intensity_field_data() -> void {
    auto wl = ValueType(data_["intensity"]["wl"]);
    auto nEig = int(data_["intensity"]["n_eig"]);
    auto centroid = std::vector<ValueType>(data_["intensity"]["centroid"].begin(),
                                     data_["intensity"]["centroid"].end());

    for (const auto& intData : data_["intensity"]["data"]) {
      auto m = intData["XYZ"][0].size();
      auto n = intData["G_real"].size();
      auto xyz = read_json_scalar_2d<ValueType>(intData["XYZ"]);
      auto w = read_json_complex_2d<ValueType>(intData["W_real"], intData["W_imag"]);
      auto s = read_json_complex_2d<ValueType>(intData["S_real"], intData["S_imag"]);
      auto vRef = read_json_complex_2d<ValueType>(intData["V_real"], intData["V_imag"]);
      auto dRef = std::vector<ValueType>(intData["D"].begin(), intData["D"].end());
      auto cIdxRef = std::vector<int>(intData["c_idx"].begin(), intData["c_idx"].end());

      std::vector<std::complex<ValueType>> v(nEig * n);
      std::vector<ValueType> d(nEig);
      std::vector<int> cIdx(nEig);

      if (std::is_same<ValueType, float>::value)
        ASSERT_EQ(bluebild_intensity_field_data_s(
                      ctx_, wl, m, n, nEig, s.data(), n, w.data(), m,
                      reinterpret_cast<float*>(xyz.data()), m, reinterpret_cast<float*>(d.data()),
                      v.data(), n, centroid.size(), reinterpret_cast<float*>(centroid.data()),
                      cIdx.data()),
                  BLUEBILD_SUCCESS);
      else
        ASSERT_EQ(bluebild_intensity_field_data_d(
                      ctx_, wl, m, n, nEig, s.data(), n, w.data(), m,
                      reinterpret_cast<double*>(xyz.data()), m, reinterpret_cast<double*>(d.data()),
                      v.data(), n, centroid.size(), reinterpret_cast<double*>(centroid.data()),
                      cIdx.data()),
                  BLUEBILD_SUCCESS);

      for (std::size_t i = 0; i < d.size(); ++i) {
        ASSERT_NEAR(d[i], dRef[i], 2.0);
      }

      for (std::size_t i = 0; i < v.size(); ++i) {
        // Eigenvectors may be mirrored
        ASSERT_NEAR(std::abs(v[i].real()), std::abs(vRef[i].real()), 1.0);
        ASSERT_NEAR(std::abs(v[i].imag()), std::abs(vRef[i].imag()), 1.0);
      }
    }
  }

  ~LofarTest() {
    bluebild_ctx_destroy(&ctx_);
  }

  const nlohmann::json& data_;
  BluebildContext ctx_;
};

using LofarSingle = LofarTest<float>;
using LofarDouble = LofarTest<double>;

TEST_P(LofarSingle, GrammMatrix) { this->gram_matrix(); }
TEST_P(LofarDouble, GrammMatrix) { this->gram_matrix(); }


TEST_P(LofarSingle, Sensitivity) { this->sensitivity_field_data(); }
TEST_P(LofarDouble, Sensitivity) { this->sensitivity_field_data(); }

TEST_P(LofarSingle, Intensity) { this->intensity_field_data(); }
TEST_P(LofarDouble, Intensity) { this->intensity_field_data(); }

static auto param_type_names(
    const ::testing::TestParamInfo<std::tuple<int, BluebildProcessingUnit>>& info) -> std::string {
  std::stringstream stream;

  if (std::get<1>(info.param) == BLUEBILD_PU_CPU) stream << "CPU_";
  else stream << "GPU_";
  stream << "nStation_" << std::get<0>(info.param);

  return stream.str();
}

#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
#define TEST_PROCESSING_UNITS BLUEBILD_PU_CPU, BLUEBILD_PU_GPU
#else
#define TEST_PROCESSING_UNITS BLUEBILD_PU_CPU
#endif

INSTANTIATE_TEST_SUITE_P(
    Lofar, LofarSingle,
    ::testing::Combine(::testing::Values(8, 24, 48),
                       ::testing::Values(TEST_PROCESSING_UNITS)),
    param_type_names);

INSTANTIATE_TEST_SUITE_P(
    Lofar, LofarDouble,
    ::testing::Combine(::testing::Values(8, 24, 48),
                       ::testing::Values(TEST_PROCESSING_UNITS)),
    param_type_names);
