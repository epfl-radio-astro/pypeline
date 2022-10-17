#include <array>
#include <optional>
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <variant>

#include "bluebild/bluebild.hpp"

using namespace bluebild;
namespace py = pybind11;

// Helper
namespace {
template <typename TARGET, typename SOURCE,
          typename = std::enable_if_t<
              std::is_integral_v<TARGET> && std::is_integral_v<SOURCE> &&
              std::is_signed_v<TARGET> && std::is_signed_v<TARGET>>>
inline auto safe_cast(SOURCE s) {
  if (s < std::numeric_limits<TARGET>::min() ||
      s > std::numeric_limits<TARGET>::max())
    throw std::overflow_error("Integer overflow due to input size or stride.");
  return static_cast<TARGET>(s);
}

template <typename T, int STYLE>
auto check_1d_array(const py::array_t<T, STYLE> &a, long shape = 0) -> void {
  if (a.ndim() != 1)
    throw InvalidParameterError();
  if (shape && a.shape(0) != shape)
    throw InvalidParameterError();
}

template <typename T, int STYLE>
auto check_2d_array(const py::array_t<T, STYLE> &a,
                    std::array<long, 2> shape = {0, 0}) -> void {
  if (a.ndim() != 2)
    throw InvalidParameterError();
  if (shape[0] && a.shape(0) != shape[0])
    throw InvalidParameterError();
  if (shape[1] && a.shape(1) != shape[1])
    throw InvalidParameterError();
}

template <typename T, int STYLE>
auto check_3d_array(const py::array_t<T, STYLE> &a,
                    std::array<long, 3> shape = {0, 0, 0}) -> void {
  if (a.ndim() != 3)
    throw InvalidParameterError();
  if (shape[0] && a.shape(0) != shape[0])
    throw InvalidParameterError();
  if (shape[1] && a.shape(1) != shape[1])
    throw InvalidParameterError();
  if (shape[2] && a.shape(2) != shape[2])
    throw InvalidParameterError();
}

template <typename T>
auto call_gram_matrix(Context &ctx,
                      const py::array_t<T, py::array::f_style> &xyz,
                      const py::array_t<std::complex<T>, py::array::f_style> &w,
                      T wl) {
  check_2d_array(w);
  check_2d_array(xyz, {w.shape(0), 3});

  auto g = py::array_t<std::complex<T>, py::array::f_style>(
      {w.shape(1), w.shape(1)});

  gram_matrix(ctx, safe_cast<int>(w.shape(0)), safe_cast<int>(w.shape(1)),
              w.data(0), safe_cast<int>(w.strides(1) / w.itemsize()),
              xyz.data(0), safe_cast<int>(xyz.strides(1) / xyz.itemsize()), wl,
              g.mutable_data(0), safe_cast<int>(g.strides(1) / g.itemsize()));

  return g;
}

template <typename T>
auto call_sensitivity_field_data(
    Context &ctx, T wl, long nEig,
    const py::array_t<T, py::array::f_style> &xyz,
    const py::array_t<std::complex<T>, py::array::f_style> &w) {
  check_2d_array(w);
  check_2d_array(xyz, {w.shape(0), 3});

  auto v = py::array_t<std::complex<T>, py::array::f_style>({w.shape(1), nEig});
  auto d = py::array_t<T>(nEig);

  sensitivity_field_data(
      ctx, wl, safe_cast<int>(w.shape(0)), safe_cast<int>(w.shape(1)),
      safe_cast<int>(nEig), w.data(0),
      safe_cast<int>(w.strides(1) / w.itemsize()), xyz.data(0),
      safe_cast<int>(xyz.strides(1) / xyz.itemsize()), d.mutable_data(0),
      v.mutable_data(0), safe_cast<int>(v.strides(1) / v.itemsize()));

  return std::make_tuple(std::move(d), std::move(v));
}

template <typename T>
auto call_intensity_field_data(
    Context &ctx, T wl, long nEig,
    const py::array_t<T, py::array::f_style> &xyz,
    const py::array_t<std::complex<T>, py::array::f_style> &w,
    const py::array_t<std::complex<T>, py::array::f_style> &s,
    const py::array_t<T, py::array::f_style> &centroids) {
  check_2d_array(w);
  check_2d_array(xyz, {w.shape(0), 3});
  check_2d_array(s, {w.shape(1), w.shape(1)});
  check_1d_array(centroids);

  auto v = py::array_t<std::complex<T>, py::array::f_style>({w.shape(1), nEig});
  auto d = py::array_t<T>(nEig);
  auto clusterIndices = py::array_t<int>(nEig);

  intensity_field_data(
      ctx, wl, safe_cast<int>(w.shape(0)), safe_cast<int>(w.shape(1)),
      safe_cast<int>(nEig), s.data(0),
      safe_cast<int>(s.strides(1) / s.itemsize()), w.data(0),
      safe_cast<int>(w.strides(1) / w.itemsize()), xyz.data(0),
      safe_cast<int>(xyz.strides(1) / xyz.itemsize()), d.mutable_data(0),
      v.mutable_data(0), safe_cast<int>(v.strides(1) / v.itemsize()),
      centroids.shape(0), centroids.data(0), clusterIndices.mutable_data(0));

  return std::make_tuple(std::move(d), std::move(v), std::move(clusterIndices));
}

template <typename T>
auto call_virtual_vis(
    Context &ctx, const py::array_t<BluebildFilter, py::array::f_style> &filter,
    const py::array_t<T, py::array::c_style> &intervals,
    const py::array_t<T, py::array::f_style> &d,
    const py::array_t<std::complex<T>, py::array::f_style> &v,
    std::optional<py::array_t<std::complex<T>, py::array::f_style>> w) {
  if(w)
    check_2d_array(w.value());
  check_1d_array(filter);
  check_2d_array(intervals, {0, 2});
  check_1d_array(d);
  check_2d_array(v);

  auto nAntenna = w ? w.value().shape(0) : v.shape(0);
  auto nBeam = w ? w.value().shape(1) : v.shape(0);
  auto nEig = v.shape(1);

  // workaround to match current style in python implementation:
  // use row major ordering and transpose inner matrices afterwards
  auto virtualVis = py::array_t<std::complex<T>, py::array::c_style>(
      {filter.size(), intervals.shape(0), nAntenna, nAntenna});

  virtual_vis<T>(
      ctx, safe_cast<int>(filter.size()), filter.data(0),
      safe_cast<int>(intervals.size() / 2), intervals.data(0),
      safe_cast<int>(intervals.strides(0) / intervals.itemsize()),
      safe_cast<int>(nEig), d.data(0), safe_cast<int>(nAntenna), v.data(0),
      safe_cast<int>(v.strides(1) / v.itemsize()), safe_cast<int>(nBeam),
      w ? w.value().data(0) : nullptr,
      w ? safe_cast<int>(w.value().strides(1) / w.value().itemsize()) : 0,
      virtualVis.mutable_data(0), virtualVis.strides(0) / virtualVis.itemsize(),
      virtualVis.strides(1) / virtualVis.itemsize(),
      virtualVis.strides(2) / virtualVis.itemsize());

  // transpose inner matrices
  auto virtualVisMut = virtualVis.template mutable_unchecked<4>();
  for (py::ssize_t i = 0; i < virtualVis.shape(0); ++i) {
    for (py::ssize_t j = 0; j < virtualVis.shape(1); ++j) {
      for (py::ssize_t k = 0; k < virtualVis.shape(2); ++k) {
        for (py::ssize_t l = 0; l < k; ++l) {
          auto tmp = virtualVisMut(i, j, k, l);
          virtualVisMut(i, j, k, l) = virtualVisMut(i, j, l, k);
          virtualVisMut(i, j, l, k) = tmp;
        }
      }
    }
  }

  return virtualVis;
}

struct SSDispatcher {
    SSDispatcher(const Context &ctx, const float wl,
                 const std::size_t Nl, const std::size_t Nh, const std::size_t Nw,
                 const py::array_t<float, pybind11::array::f_style> &pix_grid,
                 py::array_t<float, pybind11::array::f_style> &stats_std_cum,
                 py::array_t<float, pybind11::array::f_style> &stats_lsq_cum) {
        ss_ = SSf(ctx, wl, Nl, Nh, Nw, pix_grid.data(0),
                  stats_std_cum.mutable_data(0), stats_lsq_cum.mutable_data(0));
    }
    SSDispatcher(const Context &ctx, const double wl,
                 const std::size_t Nl, const std::size_t Nh, const std::size_t Nw,
                 const py::array_t<double, pybind11::array::f_style> &pix_grid,
                 py::array_t<double, pybind11::array::f_style> &stats_std_cum,
                 py::array_t<double, pybind11::array::f_style> &stats_lsq_cum) {
        ss_ = SS(ctx, wl, Nl, Nh, Nw, pix_grid.data(0),
                 stats_std_cum.mutable_data(0), stats_lsq_cum.mutable_data(0));
    }

    SSDispatcher(SSDispatcher &&) = default;
    SSDispatcher(const SSDispatcher &) = delete;
    SSDispatcher &operator=(SSDispatcher &&) = default;
    SSDispatcher &operator=(const SSDispatcher &) = delete;


    auto execute(pybind11::array d, pybind11::array v, pybind11::array xyz,
                 pybind11::array w, pybind11::array c_idx, const bool d2h) -> void {

        if (d.ndim() != 1)     throw InvalidParameterError();
        if (c_idx.ndim() != 1) throw InvalidParameterError();
        if (v.ndim() != 2)     throw InvalidParameterError();
        if (xyz.ndim() != 2)   throw InvalidParameterError();
        if (w.ndim() != 2)     throw InvalidParameterError();

        const size_t Na = xyz.shape(0);
        const size_t Ne = d.shape(0);
        const size_t Nb = v.shape(0);

        assert(c_idx.shape(0) == Ne);

        std::visit([&](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, SSf>) {
                  py::array_t<float, py::array::f_style> dArray(d);                
                  py::array_t<std::complex<float>, py::array::f_style> vArray(v);
                  py::array_t<float, py::array::f_style> xyzArray(xyz);
                  py::array_t<std::complex<float>, py::array::f_style> wArray(w);
                  py::array_t<std::size_t, py::array::f_style> cidxArray(c_idx);
                  std::get<SSf>(ss_).execute(dArray.data(0), vArray.data(0), xyzArray.data(0),
                                             wArray.data(0), cidxArray.data(0), Na, Ne, Nb, d2h);
              } else if constexpr (std::is_same_v<T, SS>) {
                  py::array_t<double, py::array::f_style> dArray(d);
                  py::array_t<std::complex<double>, py::array::f_style> vArray(v);
                  py::array_t<double, py::array::f_style> xyzArray(xyz);
                  py::array_t<std::complex<double>, py::array::f_style> wArray(w);
                  py::array_t<std::size_t, py::array::f_style> cidxArray(c_idx);
                  std::get<SS>(ss_).execute(dArray.data(0), vArray.data(0), xyzArray.data(0),
                                            wArray.data(0), cidxArray.data(0), Na, Ne, Nb, d2h);
              } else {
              throw InternalError();
          }
       }, ss_);
    }

    std::variant<std::monostate, SSf, SS> ss_;    
};

struct Nufft3d3Dispatcher {

  Nufft3d3Dispatcher(const Context &ctx, int iflag, float tol, int numTrans,
                     const py::array_t<float, pybind11::array::f_style> &x,
                     const py::array_t<float, pybind11::array::f_style> &y,
                     const py::array_t<float, pybind11::array::f_style> &z,
                     const py::array_t<float, pybind11::array::f_style> &s,
                     const py::array_t<float, pybind11::array::f_style> &t,
                     const py::array_t<float, pybind11::array::f_style> &u)
      : m_(safe_cast<int>(x.shape(0))), n_(safe_cast<int>(s.shape(0))),
        numTrans_(numTrans) {
    check_1d_array(x);
    check_1d_array(y, x.shape(0));
    check_1d_array(z, x.shape(0));
    check_1d_array(s);
    check_1d_array(t, s.shape(0));
    check_1d_array(u, s.shape(0));
    plan_ = Nufft3d3f(ctx, iflag, tol, numTrans, m_, x.data(0), y.data(0),
                      z.data(0), n_, s.data(0), t.data(0), u.data(0));
  }

  Nufft3d3Dispatcher(const Context &ctx, int iflag, double tol, int numTrans,
                     const py::array_t<double, pybind11::array::f_style> &x,
                     const py::array_t<double, pybind11::array::f_style> &y,
                     const py::array_t<double, pybind11::array::f_style> &z,
                     const py::array_t<double, pybind11::array::f_style> &s,
                     const py::array_t<double, pybind11::array::f_style> &t,
                     const py::array_t<double, pybind11::array::f_style> &u)
      : m_(safe_cast<int>(x.shape(0))), n_(s.shape(0)),
        numTrans_(numTrans) {
    check_1d_array(x);
    check_1d_array(y, x.shape(0));
    check_1d_array(z, x.shape(0));
    check_1d_array(s);
    check_1d_array(t, s.shape(0));
    check_1d_array(u, s.shape(0));
    plan_ = Nufft3d3(ctx, iflag, tol, numTrans, m_, x.data(0), y.data(0),
                     z.data(0), n_, s.data(0), t.data(0), u.data(0));
  }

  Nufft3d3Dispatcher(Nufft3d3Dispatcher &&) = default;

  Nufft3d3Dispatcher(const Nufft3d3Dispatcher &) = delete;

  Nufft3d3Dispatcher &operator=(Nufft3d3Dispatcher &&) = default;

  Nufft3d3Dispatcher &operator=(const Nufft3d3Dispatcher &) = delete;

  auto execute(pybind11::array cj) -> pybind11::array {
    if (cj.ndim() > 2 || cj.size() != m_ * numTrans_)
      throw InvalidParameterError();
    return std::visit(
        [&](auto &&arg) -> pybind11::array {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, Nufft3d3f>) {
            py::array_t<std::complex<float>, py::array::f_style> cjArray(
                cj.reshape({m_ * numTrans_}));
            check_1d_array(cjArray, m_);
            // use c_style 1d output, to allow for reshaping afterwards with
            // numTrans_ as first dimension
            py::array_t<std::complex<float>, py::array::c_style> fkArray(
                n_ * numTrans_);

            std::get<Nufft3d3f>(plan_).execute(cjArray.data(0),
                                               fkArray.mutable_data(0));

            if (numTrans_ > 1)  fkArray = fkArray.reshape({numTrans_, n_});
            return fkArray;
          } else if constexpr (std::is_same_v<T, Nufft3d3>) {
            py::array_t<std::complex<double>, py::array::f_style> cjArray(
                cj.reshape({m_ * numTrans_}));
            check_1d_array(cjArray, m_);
            // use c_style 1d output, to allow for reshaping afterwards with
            // numTrans_ as first dimension
            py::array_t<std::complex<double>, py::array::c_style> fkArray(
                n_ * numTrans_);

            std::get<Nufft3d3>(plan_).execute(cjArray.data(0),
                                              fkArray.mutable_data(0));

            if (numTrans_ > 1)  fkArray = fkArray.reshape({numTrans_, n_});
            return fkArray;
          } else {
            throw InternalError();
            return py::array_t<std::complex<double>, py::array::f_style>();
          }
        },
        plan_);
  }

  int m_, n_, numTrans_;
  std::variant<std::monostate, Nufft3d3f, Nufft3d3> plan_;
};


struct PeriodicSynthesisDispatcher {

  PeriodicSynthesisDispatcher(
      Context &ctx, int nAntenna, int nBeam, int nIntervals,
      const py::array_t<BluebildFilter, pybind11::array::f_style> &filter,
      const py::array &lmnX, const py::array &lmnY, const py::array &lmnZ,
      const std::string &precision, double tol)
      : nIntervals_(nIntervals), nPixel_(lmnX.shape(0)) {
    if (precision == "single" || precision == "SINGLE") {
      py::array_t<float, pybind11::array::f_style | py::array::forcecast>
          lmnXArray(lmnX);
      py::array_t<float, pybind11::array::f_style | py::array::forcecast>
          lmnYArray(lmnY);
      py::array_t<float, pybind11::array::f_style | py::array::forcecast>
          lmnZArray(lmnZ);
      check_1d_array(filter);
      check_1d_array(lmnXArray);
      check_1d_array(lmnYArray, lmnXArray.shape(0));
      check_1d_array(lmnZArray, lmnXArray.shape(0));
      plan_ = PeriodicSynthesis<float>(ctx, tol, nAntenna, nBeam, nIntervals,
                                       filter.shape(0), filter.data(0),
                                       lmnXArray.shape(0), lmnXArray.data(0),
                                       lmnYArray.data(0), lmnZArray.data(0));
    } else if (precision == "double" || precision == "DOUBLE") {
      py::array_t<double, pybind11::array::f_style | py::array::forcecast>
          lmnXArray(lmnX);
      py::array_t<double, pybind11::array::f_style | py::array::forcecast>
          lmnYArray(lmnY);
      py::array_t<double, pybind11::array::f_style | py::array::forcecast>
          lmnZArray(lmnZ);
      check_1d_array(filter);
      check_1d_array(lmnXArray);
      check_1d_array(lmnYArray, lmnXArray.shape(0));
      check_1d_array(lmnZArray, lmnXArray.shape(0));
      plan_ = PeriodicSynthesis<double>(ctx, tol, nAntenna, nBeam, nIntervals,
                                       filter.shape(0), filter.data(0),
                                       lmnXArray.shape(0), lmnXArray.data(0),
                                       lmnYArray.data(0), lmnZArray.data(0));
    } else {
      throw InvalidParameterError();
    }
  }

  PeriodicSynthesisDispatcher(PeriodicSynthesisDispatcher &&) = default;

  PeriodicSynthesisDispatcher(const PeriodicSynthesisDispatcher &) = delete;

  PeriodicSynthesisDispatcher &
  operator=(PeriodicSynthesisDispatcher &&) = default;

  PeriodicSynthesisDispatcher &
  operator=(const PeriodicSynthesisDispatcher &) = delete;

  auto collect(int nEig, double wl, pybind11::array intervals,
               pybind11::array w, pybind11::array xyz, pybind11::array uvwX,
               pybind11::array uvwY, pybind11::array uvwZ,
               std::optional<pybind11::array> s) -> void {
    std::visit(
        [&](auto &&arg) -> void {
          using variantType = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<variantType, PeriodicSynthesis<float>> ||
                        std::is_same_v<variantType,
                                       PeriodicSynthesis<double>>) {
            using T = typename variantType::valueType;
            py::array_t<T, py::array::c_style | py::array::forcecast> intervalsArray(intervals);
            check_2d_array(intervalsArray, {nIntervals_, 2});
            py::array_t<std::complex<T>,
                        py::array::f_style | py::array::forcecast>
                wArray(w);
            check_2d_array(wArray);
            auto nAntenna = wArray.shape(0);
            auto nBeam = wArray.shape(1);
            py::array_t<T, py::array::f_style | py::array::forcecast> xyzArray(
                xyz);
            check_2d_array(xyzArray, {nAntenna, 3});
            py::array_t<T, py::array::f_style | py::array::forcecast> uvwXArray(
                uvwX);
            check_1d_array(uvwXArray, nAntenna * nAntenna);
            py::array_t<T, py::array::f_style | py::array::forcecast> uvwYArray(
                uvwY);
            check_1d_array(uvwYArray, nAntenna * nAntenna);
            py::array_t<T, py::array::f_style | py::array::forcecast> uvwZArray(
                uvwZ);
            check_1d_array(uvwZArray, nAntenna * nAntenna);
            std::optional<py::array_t<
                std::complex<T>, py::array::f_style | py::array::forcecast>>
                sArray;
            if (s) {
              sArray = py::array_t<std::complex<T>,
                                   py::array::f_style | py::array::forcecast>(
                  s.value());
              check_2d_array(sArray.value(), {nBeam, nBeam});
            }
            std::get<PeriodicSynthesis<T>>(plan_).collect(
                nEig, wl, intervalsArray.data(0),
                safe_cast<int>(intervals.strides(0) / intervals.itemsize()),
                s ? sArray.value().data(0) : nullptr,
                s ? safe_cast<int>(sArray.value().strides(1) /
                                   sArray.value().itemsize())
                  : 0,
                wArray.data(0),
                safe_cast<int>(wArray.strides(1) / wArray.itemsize()),
                xyzArray.data(0),
                safe_cast<int>(xyzArray.strides(1) / xyzArray.itemsize()),
                uvwXArray.data(0), uvwYArray.data(0), uvwZArray.data(0));

          } else {
            throw InternalError();
          }
        },
        plan_);
  }

  auto get(BluebildFilter f) -> py::array {
    return std::visit(
        [&](auto &&arg) -> pybind11::array {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, PeriodicSynthesis<double>>) {
            py::array_t<double> out({nIntervals_, nPixel_});
            std::get<PeriodicSynthesis<double>>(plan_).get(
                f, out.mutable_data(0),
                safe_cast<int>(out.strides(0) / out.itemsize()));
            return out;
          } else if constexpr (std::is_same_v<T, PeriodicSynthesis<float>>) {
            py::array_t<float> out({nIntervals_, nPixel_});
            std::get<PeriodicSynthesis<float>>(plan_).get(
                f, out.mutable_data(0),
                safe_cast<int>(out.strides(0) / out.itemsize()));
            return out;
          } else {
            throw InternalError();
            return py::array_t<double, py::array::f_style>();
          }
        },
        plan_);
  }

  std::variant<std::monostate, PeriodicSynthesis<float>,
               PeriodicSynthesis<double>>
      plan_;
  int nIntervals_, nPixel_;
};




} // namespace

// Create module

// NOTE: Function overloading does NOT generally try to find the best matching
// types and the declaration order matters. Always declare single precision
// functions first, as otherwise double precision versions will always be
// selected.
PYBIND11_MODULE(pybluebild, m) {
  m.doc() = R"pbdoc(
        Bluebild
    )pbdoc";
#ifdef BLUEBILD_VERSION
  m.attr("__version__") = BLUEBILD_VERSION;
#else
  m.attr("__version__") = "dev";
#endif

  py::enum_<BluebildProcessingUnit>(m, "ProcessingUnit")
      .value("AUTO", BLUEBILD_PU_AUTO)
      .value("CPU", BLUEBILD_PU_CPU)
      .value("GPU", BLUEBILD_PU_GPU);

  py::enum_<BluebildFilter>(m, "Filter")
      .value("LSQ", BLUEBILD_FILTER_LSQ)
      .value("STD", BLUEBILD_FILTER_STD)
      .value("SQRT", BLUEBILD_FILTER_SQRT)
      .value("INV", BLUEBILD_FILTER_INV);

  pybind11::class_<Context>(m, "Context")
      .def(pybind11::init<BluebildProcessingUnit>(),
           pybind11::arg("pu").noconvert())
      .def("processing_unit", &Context::processing_unit)
      .def(
          "gram_matrix",
          [](Context &ctx,
             const py::array_t<float, pybind11::array::f_style> &xyz,
             const py::array_t<std::complex<float>, pybind11::array::f_style>
                 &w,
             float wl) { return call_gram_matrix(ctx, xyz, w, wl); },
          pybind11::arg("XYZ"), pybind11::arg("W"), pybind11::arg("wl"))
      .def(
          "gram_matrix",
          [](Context &ctx,
             const py::array_t<double, pybind11::array::f_style> &xyz,
             const py::array_t<std::complex<double>, pybind11::array::f_style>
                 &w,
             double wl) { return call_gram_matrix(ctx, xyz, w, wl); },
          pybind11::arg("XYZ"), pybind11::arg("W"), pybind11::arg("wl"))
      .def(
          "sensitivity_field_data",
          [](Context &ctx, long nEig,
             const py::array_t<float, pybind11::array::f_style> &xyz,
             const py::array_t<std::complex<float>, pybind11::array::f_style>
                 &w,
             float wl) {
            return call_sensitivity_field_data(ctx, wl, nEig, xyz, w);
          },
          pybind11::arg("n_eig"), pybind11::arg("XYZ"), pybind11::arg("W"),
          pybind11::arg("wl"))
      .def(
          "sensitivity_field_data",
          [](Context &ctx, long nEig,
             const py::array_t<double, pybind11::array::f_style> &xyz,
             const py::array_t<std::complex<double>, pybind11::array::f_style>
                 &w,
             double wl) {
            return call_sensitivity_field_data(ctx, wl, nEig, xyz, w);
          },
          pybind11::arg("n_eig"), pybind11::arg("XYZ"), pybind11::arg("W"),
          pybind11::arg("wl"))
      .def(
          "intensity_field_data",
          [](Context &ctx, long nEig,
             const py::array_t<float, pybind11::array::f_style> &xyz,
             const py::array_t<std::complex<float>, pybind11::array::f_style>
                 &w,
             float wl,
             const py::array_t<std::complex<float>, pybind11::array::f_style>
                 &s,
             const py::array_t<float, pybind11::array::f_style> &centroids) {
            return call_intensity_field_data(ctx, wl, nEig, xyz, w, s,
                                             centroids);
          },
          pybind11::arg("n_eig"), pybind11::arg("XYZ"), pybind11::arg("W"),
          pybind11::arg("wl"), pybind11::arg("S"), pybind11::arg("centroids"))
      .def(
          "intensity_field_data",
          [](Context &ctx, long nEig,
             const py::array_t<double, pybind11::array::f_style> &xyz,
             const py::array_t<std::complex<double>, pybind11::array::f_style>
                 &w,
             double wl,
             const py::array_t<std::complex<double>, pybind11::array::f_style>
                 &s,
             const py::array_t<double, pybind11::array::f_style> &centroids) {
            return call_intensity_field_data(ctx, wl, nEig, xyz, w, s,
                                             centroids);
          },
          pybind11::arg("n_eig"), pybind11::arg("XYZ"), pybind11::arg("W"),
          pybind11::arg("wl"), pybind11::arg("S"), pybind11::arg("centroids"))
      .def(
          "virtual_vis",
          [](Context &ctx,
             const py::array_t<BluebildFilter, py::array::f_style> &filter,
             const py::array_t<float, py::array::c_style> &intervals,
             const py::array_t<float, py::array::f_style> &d,
             const py::array_t<std::complex<float>, py::array::f_style> &v,
             std::optional<py::array_t<std::complex<float>, py::array::f_style>> w) {
            return call_virtual_vis(ctx, filter, intervals, d, v, w);
          },
          pybind11::arg("filter"), pybind11::arg("intervals"),
          pybind11::arg("d"), pybind11::arg("v"), pybind11::arg("w"))
      .def(
          "virtual_vis",
          [](Context &ctx,
             const py::array_t<BluebildFilter, py::array::f_style> &filter,
             const py::array_t<double, py::array::c_style> &intervals,
             const py::array_t<double, py::array::f_style> &d,
             const py::array_t<std::complex<double>, py::array::f_style> &v,
             std::optional<py::array_t<std::complex<double>, py::array::f_style>> w) {
            return call_virtual_vis(ctx, filter, intervals, d, v, w);
          },
          pybind11::arg("filter"), pybind11::arg("intervals"),
          pybind11::arg("d"), pybind11::arg("v"), pybind11::arg("w"));

  pybind11::class_<Nufft3d3Dispatcher>(m, "Nufft3d3")
      .def(pybind11::init<
               const Context &, int, float, int,
               const py::array_t<float, pybind11::array::f_style> &,
               const py::array_t<float, pybind11::array::f_style> &,
               const py::array_t<float, pybind11::array::f_style> &,
               const py::array_t<float, pybind11::array::f_style> &,
               const py::array_t<float, pybind11::array::f_style> &,
               const py::array_t<float, pybind11::array::f_style> &>(),
           pybind11::arg("ctx"), pybind11::arg("iflag"), pybind11::arg("tol"),
           pybind11::arg("num_trans"), pybind11::arg("x"), pybind11::arg("y"),
           pybind11::arg("z"), pybind11::arg("s"), pybind11::arg("t"),
           pybind11::arg("u"))
      .def(pybind11::init<
               const Context &, int, double, int,
               const py::array_t<double, pybind11::array::f_style> &,
               const py::array_t<double, pybind11::array::f_style> &,
               const py::array_t<double, pybind11::array::f_style> &,
               const py::array_t<double, pybind11::array::f_style> &,
               const py::array_t<double, pybind11::array::f_style> &,
               const py::array_t<double, pybind11::array::f_style> &>(),
           pybind11::arg("ctx"), pybind11::arg("iflag"), pybind11::arg("tol"),
           pybind11::arg("num_trans"), pybind11::arg("x"), pybind11::arg("y"),
           pybind11::arg("z"), pybind11::arg("s"), pybind11::arg("t"),
           pybind11::arg("u"))
      .def("execute", &Nufft3d3Dispatcher::execute, pybind11::arg("cj"));

  pybind11::class_<PeriodicSynthesisDispatcher>(m, "PeriodicSynthesis")
      .def(pybind11::init<
               Context &, int, int, int,
               const py::array_t<BluebildFilter, pybind11::array::f_style> &,
               const py::array &, const py::array &, const py::array &,
               const std::string &, double>(),
           pybind11::arg("ctx"), pybind11::arg("n_antenna"),
           pybind11::arg("n_beam"), pybind11::arg("n_intervals"),
           pybind11::arg("filter"), pybind11::arg("lmn_x"),
           pybind11::arg("lmn_y"), pybind11::arg("lmn_y"),
           pybind11::arg("precision"), pybind11::arg("tol"))
      .def("collect", &PeriodicSynthesisDispatcher::collect,
           pybind11::arg("nEig"), pybind11::arg("wl"),
           pybind11::arg("intervals"), pybind11::arg("w"), pybind11::arg("xyz"),
           pybind11::arg("uvwX"), pybind11::arg("uvwY"), pybind11::arg("uvwY"),
           pybind11::arg("s"))
      .def("get", &PeriodicSynthesisDispatcher::get, pybind11::arg("f"));

  pybind11::class_<SSDispatcher>(m, "SS")
      .def(pybind11::init<
           const Context &, const float,
           const std::size_t, const std::size_t, const std::size_t,
           const py::array_t<float, pybind11::array::f_style> &,
           py::array_t<float, pybind11::array::f_style> &,
           py::array_t<float, pybind11::array::f_style> &>(),
           pybind11::arg("ctx"), pybind11::arg("wl"),
           pybind11::arg("Nl"), pybind11::arg("Nh"), pybind11::arg("Nw"),
           pybind11::arg("pix_grid"),
           pybind11::arg("stats_std_cum"), pybind11::arg("stats_lsq_cum"))
      .def(pybind11::init<
           const Context &, const double,
           const std::size_t, const std::size_t, const std::size_t,
           const py::array_t<double, pybind11::array::f_style> &,
           py::array_t<double, pybind11::array::f_style> &,
           py::array_t<double, pybind11::array::f_style> &>(),
           pybind11::arg("ctx"), pybind11::arg("wl"),
           pybind11::arg("Nl"), pybind11::arg("Nh"), pybind11::arg("Nw"),
           pybind11::arg("pix_grid"),
           pybind11::arg("stats_std_cum"), pybind11::arg("stats_lsq_cum"))
      .def("execute", &SSDispatcher::execute,
           pybind11::arg("d"), pybind11::arg("v"),
           pybind11::arg("xyz"), pybind11::arg("w"), pybind11::arg("c_idx"),
           pybind11::arg("d2h"));
}
