#include <cmath>
#include <complex>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <fstream>
#include <iostream>
#include <typeinfo>
#include <string>

#include "bluebild/bluebild.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"


static auto get_lofar_json(std::string json) -> nlohmann::json {
  const std::string fjson = std::string(BLUEBILD_TEST_DATA_DIR) + "/" + json + ".json";
  //std::cout << fjson << std::endl;
  std::ifstream file(fjson);
  nlohmann::json j;
  file >> j;
  return j;
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

template <typename T, typename JSON>
static auto read_json_scalar_3d(JSON j) -> std::vector<T> {
  std::vector<T> w;

  for (auto &slice : j) {
    for (auto &col : slice) {
      for (auto &val : col) {
        w.emplace_back(val);
      }
    }
  }

  return w;
}

template <typename T>
static auto compute_rmse(std::vector<T> vsol, std::vector<T> vref) -> T {
  assert(vsol.size() == vref.size());
  size_t N = vref.size();
  auto rmse = 0.0;
  for (size_t i = 0; i<vref.size(); i++) {
    rmse += (vsol[i] - vref[i]) * (vsol[i] - vref[i]);
  }
  return sqrt(rmse / static_cast<T>(N));
}

template <typename T>
class LofarSSTest : public ::testing::TestWithParam<std::tuple<std::string, BluebildProcessingUnit>> {
protected:
  using ValueType = T;

  LofarSSTest() {
    if (bluebild_ctx_create(std::get<1>(GetParam()), &ctx_) != BLUEBILD_SUCCESS)
      throw std::runtime_error("Failed to create context.");
  }

  auto standard_synthesizer() -> void {

    nlohmann::json data_ = get_lofar_json(std::get<0>(GetParam()));

    auto wl = ValueType(data_["wl"]);    
    auto Na = size_t(data_["Na"]);
    auto Nb = size_t(data_["Nb"]);
    auto Nc = size_t(data_["Nc"]);
    auto Ne = size_t(data_["Ne"]);
    auto Nh = size_t(data_["Nh"]);
    auto Nl = size_t(data_["Nl"]);
    auto Nw = size_t(data_["Nw"]);
    auto grid = read_json_scalar_3d<ValueType>(data_["grid"]);
    auto stats_std_ref = read_json_scalar_3d<ValueType>(data_["statistics_std"]);

    std::vector<ValueType> stats_std(Nl * Nh * Nw);
    std::vector<ValueType> stats_lsq(Nl * Nh * Nw);
    std::vector<ValueType> stats_std_cum(Nl * Nh * Nw, 0.0);
    std::vector<ValueType> stats_lsq_cum(Nl * Nh * Nw, 0.0);

    for (int epoch = 0; epoch < data_["XYZ"].size(); ++epoch) {

        auto xyz   = read_json_scalar_2d<ValueType>(data_["XYZ"][epoch]);
        auto w     = read_json_complex_2d<ValueType>(data_["W_real"][epoch], data_["W_imag"][epoch]);
        auto d     = read_json_scalar_2d<ValueType>(data_["D"][epoch]);
        auto v     = read_json_complex_2d<ValueType>(data_["V_real"][epoch], data_["V_imag"][epoch]);
        auto c_idx = std::vector<size_t>(data_["c_idx"][epoch].begin(), data_["c_idx"][epoch].end());
       
        if (std::is_same<ValueType, float>::value) {
          ASSERT_EQ(
            bluebild_standard_synthesizer_s(ctx_,
                                            reinterpret_cast<float*>(d.data()),
                                            v.data(),
                                            reinterpret_cast<float*>(xyz.data()),
                                            reinterpret_cast<std::complex<float>*>(w.data()),
                                            c_idx.data(),
                                            Nl,
                                            reinterpret_cast<float*>(grid.data()),
                                            wl, Na, Nb, Nc, Ne, Nh, Nw,
                                            reinterpret_cast<float*>(stats_std.data()),
                                            reinterpret_cast<float*>(stats_lsq.data()),
                                            reinterpret_cast<float*>(stats_std_cum.data()),
                                            reinterpret_cast<float*>(stats_lsq_cum.data())),
            BLUEBILD_SUCCESS);
        }
        else {
          ASSERT_EQ(
            bluebild_standard_synthesizer_d(ctx_,
                                            reinterpret_cast<double*>(d.data()),
                                            v.data(),
                                            reinterpret_cast<double*>(xyz.data()),
                                            reinterpret_cast<std::complex<double>*>(w.data()),
                                            c_idx.data(),
                                            Nl,
                                            reinterpret_cast<double*>(grid.data()),
                                            wl, Na, Nb, Nc, Ne, Nh, Nw,
                                            reinterpret_cast<double*>(stats_std.data()),
                                            reinterpret_cast<double*>(stats_lsq.data()),
                                            reinterpret_cast<double*>(stats_std_cum.data()),
                                            reinterpret_cast<double*>(stats_lsq_cum.data())),
            BLUEBILD_SUCCESS);
        }
    }

    // Compute RMSE between cumulated solution and reference (saved in json from ref Python)
    auto rmse = compute_rmse(stats_std_cum, stats_std_ref);
    char buffer[50];
    sprintf(buffer, "Got rmse = %.5e", rmse);
    if (std::is_same<ValueType, double>::value) {
      ASSERT_TRUE(rmse < 1.0E-8) << buffer;
    } else if (std::is_same<ValueType, float>::value) {
      ASSERT_TRUE(rmse < 1.0E-3) <<  buffer;
    }
  }

  ~LofarSSTest() {
    bluebild_ctx_destroy(&ctx_);
  }

  //const nlohmann::json& data_;
  BluebildContext ctx_;
};

using LofarSSSingle = LofarSSTest<float>;
using LofarSSDouble = LofarSSTest<double>;

TEST_P(LofarSSSingle, StandardSynthesizer) { this->standard_synthesizer(); }
TEST_P(LofarSSDouble, StandardSynthesizer) { this->standard_synthesizer(); }

static auto param_type_names(
    const ::testing::TestParamInfo<std::tuple<std::string, BluebildProcessingUnit>>& info) -> std::string {
  std::stringstream stream;

  if (std::get<1>(info.param) == BLUEBILD_PU_CPU) stream << "CPU_";
  else stream << "GPU_";
  stream << "jsonFile_" << std::get<0>(info.param);

  return stream.str();
}

#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
#define TEST_PROCESSING_UNITS BLUEBILD_PU_CPU, BLUEBILD_PU_GPU
#else
#define TEST_PROCESSING_UNITS BLUEBILD_PU_CPU
#endif

INSTANTIATE_TEST_SUITE_P(
    Lofar, LofarSSSingle,
    ::testing::Combine(::testing::Values("lofar_ss_32"),
                       ::testing::Values(TEST_PROCESSING_UNITS)),
    param_type_names);

INSTANTIATE_TEST_SUITE_P(
    Lofar, LofarSSDouble,
    ::testing::Combine(::testing::Values("lofar_ss_64"),
                       ::testing::Values(TEST_PROCESSING_UNITS)),
    param_type_names);
