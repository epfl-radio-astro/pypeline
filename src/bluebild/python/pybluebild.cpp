#include <array>
#include <iostream>
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
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
          pybind11::arg("wl"), pybind11::arg("S"), pybind11::arg("centroids"));


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
}
