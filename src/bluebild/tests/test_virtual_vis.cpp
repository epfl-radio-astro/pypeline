#include <cmath>
#include <complex>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <fstream>

#include "bluebild/bluebild.h"
#include "bluebild/bluebild.hpp"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"



static auto get_virt_vis_json() -> const nlohmann::json& {
  static nlohmann::json data = []() {
    std::ifstream file(std::string(BLUEBILD_TEST_DATA_DIR) + "/data_virt_vis.json");
    nlohmann::json j;
    file >> j;
    return j;
  }();

  return data;
}

template <typename T, typename JSON>
static auto read_json_complex_1d(JSON jReal, JSON jImag) -> std::vector<std::complex<T>> {
  std::vector<std::complex<T>> w;


  auto real = jReal.begin();
  auto imag = jImag.begin();
  for (; real != jReal.end(); ++real, ++imag) {
    w.emplace_back(*real, *imag);
  }
  return w;
}

static auto filter_name(BluebildFilter f) -> const char* {
  switch(f) {
    case BLUEBILD_FILTER_LSQ: return "lsq";
    case BLUEBILD_FILTER_STD: return "std";
    case BLUEBILD_FILTER_SQRT: return "sqrt";
    case BLUEBILD_FILTER_INV: return "inv";
  }
  return "error";
}

template <typename T>
class VirtualVisTest : public ::testing::TestWithParam<std::tuple<BluebildFilter, BluebildProcessingUnit>> {
protected:
  using ValueType = T;

  VirtualVisTest() : ctx_(std::get<1>(GetParam())) {}

  auto test(bool unbeam) -> void {
    const auto &data =
        get_virt_vis_json()[std::string(filter_name(std::get<0>(GetParam()))) + (unbeam ? "" : "_no_w")];
    const auto filter = std::get<0>(GetParam());

    const int nAntenna = data["N_antenna"];
    const int nBeam = data["N_beam"];
    const int nEig = data["N_eig"];
    const auto D = data["D"].get<std::vector<T>>();
    const auto intervals = data["intervals"].get<std::vector<T>>();
    const auto V = read_json_complex_1d<T>(data["V_real"], data["V_imag"]);
    const auto W = unbeam ? read_json_complex_1d<T>(data["W_real"], data["W_imag"]) : std::vector<std::complex<T>>();
    const auto virtVisRef =
        read_json_complex_1d<T>(data["virt_vis_real"], data["virt_vis_imag"]);

    std::vector<std::complex<T>> virtVis(virtVisRef.size());

    bluebild::virtual_vis(ctx_, 1, &filter, intervals.size() / 2,
                          intervals.data(), 2, nEig, D.data(), nAntenna,
                          V.data(), unbeam ? nBeam : nAntenna, nBeam, W.data(),
                          nAntenna, virtVis.data(),
                          (intervals.size() / 2) * nAntenna * nAntenna,
                          nAntenna * nAntenna, nAntenna);

    for (std::size_t i = 0; i < virtVis.size(); ++i) {
      ASSERT_NEAR(virtVis[i].real(), virtVisRef[i].real(), 1e-3);
      ASSERT_NEAR(virtVis[i].imag(), virtVisRef[i].imag(), 1e-3);
    }
  }

  bluebild::Context ctx_;
};

using VirtualVisUnbeamSingle = VirtualVisTest<float>;
using VirtualVisUnbeamDouble = VirtualVisTest<double>;

TEST_P(VirtualVisUnbeamSingle, With_W) { this->test(true); }
TEST_P(VirtualVisUnbeamDouble, With_W) { this->test(true); }

using VirtualVisSingle = VirtualVisTest<float>;
using VirtualVisDouble = VirtualVisTest<double>;

TEST_P(VirtualVisSingle, Without_W) { this->test(false); }
TEST_P(VirtualVisDouble, Without_W) { this->test(false); }

static auto param_type_names(
    const ::testing::TestParamInfo<std::tuple<BluebildFilter, BluebildProcessingUnit>>& info) -> std::string {
  std::stringstream stream;

  if (std::get<1>(info.param) == BLUEBILD_PU_CPU) stream << "CPU_";
  else stream << "GPU_";
  stream << filter_name(std::get<0>(info.param));

  return stream.str();
}

#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
#define TEST_PROCESSING_UNITS BLUEBILD_PU_CPU, BLUEBILD_PU_GPU
#else
#define TEST_PROCESSING_UNITS BLUEBILD_PU_CPU
#endif

INSTANTIATE_TEST_SUITE_P(
    Lofar, VirtualVisUnbeamSingle,
    ::testing::Combine(::testing::Values(BLUEBILD_FILTER_LSQ,
                                         BLUEBILD_FILTER_STD,
                                         BLUEBILD_FILTER_SQRT,
                                         BLUEBILD_FILTER_INV),
                       ::testing::Values(TEST_PROCESSING_UNITS)),
    param_type_names);

INSTANTIATE_TEST_SUITE_P(
    Lofar, VirtualVisUnbeamDouble,
    ::testing::Combine(::testing::Values(BLUEBILD_FILTER_LSQ,
                                         BLUEBILD_FILTER_STD,
                                         BLUEBILD_FILTER_SQRT,
                                         BLUEBILD_FILTER_INV),
                       ::testing::Values(TEST_PROCESSING_UNITS)),
    param_type_names);

INSTANTIATE_TEST_SUITE_P(
    Lofar, VirtualVisSingle,
    ::testing::Combine(::testing::Values(BLUEBILD_FILTER_LSQ),
                       ::testing::Values(TEST_PROCESSING_UNITS)),
    param_type_names);

INSTANTIATE_TEST_SUITE_P(
    Lofar, VirtualVisDouble,
    ::testing::Combine(::testing::Values(BLUEBILD_FILTER_LSQ),
                       ::testing::Values(TEST_PROCESSING_UNITS)),
    param_type_names);
