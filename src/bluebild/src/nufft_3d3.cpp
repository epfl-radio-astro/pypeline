#include <complex>
#include <optional>

#include "bluebild/config.h"
#include "bluebild/exceptions.hpp"
#include "bluebild/nufft_3d3.hpp"

#include "host/nufft_3d3_host.hpp"
#include "context_internal.hpp"

#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
#include "memory/buffer.hpp"
#include "gpu/nufft_3d3_gpu.hpp"
#include "gpu/util/gpu_runtime_api.hpp"
#include "gpu/util/gpu_util.hpp"
#endif

namespace bluebild {

template <typename T> struct Nufft3d3Internal {
  Nufft3d3Internal(std::shared_ptr<ContextInternal> ctx, int iflag, T tol,
                   int numTrans, int M, const T *x, const T *y, const T *z,
                   int N, const T *s, const T *t, const T *u)
      : M_(M), N_(N), ctx_(std::move(ctx)) {
    if (ctx_->processing_unit() == BLUEBILD_PU_CPU) {
      planHost_ = Nufft3d3Host<T>(iflag, tol, numTrans, M, x, y, z, N, s, t, u);
    } else {
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
      BufferType<T> xBuffer, yBuffer, zBuffer, sBuffer,
          tBuffer, uBuffer;
      auto xDevice = x;
      auto yDevice = y;
      auto zDevice = z;
      auto sDevice = s;
      auto tDevice = t;
      auto uDevice = u;

      if (!is_device_ptr(x)) {
        xBuffer = create_buffer<T>(ctx_->allocators().gpu(), M);
        xDevice = xBuffer.get();
        gpu::check_status(gpu::memcpy_async(xBuffer.get(), x, M * sizeof(T),
                                            gpu::flag::MemcpyHostToDevice,
                                            ctx_->gpu_stream()));
      }
      if (!is_device_ptr(y)) {
        yBuffer = create_buffer<T>(ctx_->allocators().gpu(), M);
        yDevice = yBuffer.get();
        gpu::check_status(gpu::memcpy_async(yBuffer.get(), y, M * sizeof(T),
                                            gpu::flag::MemcpyHostToDevice,
                                            ctx_->gpu_stream()));
      }
      if (!is_device_ptr(z)) {
        zBuffer = create_buffer<T>(ctx_->allocators().gpu(), M);
        zDevice = zBuffer.get();
        gpu::check_status(gpu::memcpy_async(zBuffer.get(), z, M * sizeof(T),
                                            gpu::flag::MemcpyHostToDevice,
                                            ctx_->gpu_stream()));
      }
      if (!is_device_ptr(s)) {
        sBuffer = create_buffer<T>(ctx_->allocators().gpu(), N);
        sDevice = sBuffer.get();
        gpu::check_status(gpu::memcpy_async(sBuffer.get(), s, N * sizeof(T),
                                            gpu::flag::MemcpyHostToDevice,
                                            ctx_->gpu_stream()));
      }
      if (!is_device_ptr(t)) {
        tBuffer = create_buffer<T>(ctx_->allocators().gpu(), N);
        tDevice = tBuffer.get();
        gpu::check_status(gpu::memcpy_async(tBuffer.get(), t, N * sizeof(T),
                                            gpu::flag::MemcpyHostToDevice,
                                            ctx_->gpu_stream()));
      }
      if (!is_device_ptr(u)) {
        uBuffer = create_buffer<T>(ctx_->allocators().gpu(), N);
        uDevice = uBuffer.get();
        gpu::check_status(gpu::memcpy_async(uBuffer.get(), u, N * sizeof(T),
                                            gpu::flag::MemcpyHostToDevice,
                                            ctx_->gpu_stream()));
      }

      gpu::stream_synchronize(ctx_->gpu_stream()); // cufinufft cannot be asigned a stream
      planGPU_ = Nufft3d3GPU<T>(iflag, tol, numTrans, M, xDevice, yDevice, zDevice, N, sDevice, tDevice, uDevice);
      gpu::stream_synchronize(nullptr);
#else
      throw GPUSupportError();
#endif
    }
  }

  void exec(const std::complex<T> *cj, std::complex<T> *fk) {
    if (ctx_->processing_unit() == BLUEBILD_PU_CPU) {
      planHost_.value().execute(cj, fk);
    } else {
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
      BufferType<gpu::ComplexType<T>> cjBuffer, fkBuffer;
      auto cjDevice = reinterpret_cast<const gpu::ComplexType<T> *>(cj);
      auto fkDevice = reinterpret_cast<gpu::ComplexType<T> *>(fk);

      if (!is_device_ptr(cj)) {
        cjBuffer = create_buffer<gpu::ComplexType<T>>(ctx_->allocators().gpu(), M_);
        cjDevice = cjBuffer.get();
        gpu::check_status(gpu::memcpy_async(cjBuffer.get(), cj, M_ * sizeof(gpu::ComplexType<T>),
                                            gpu::flag::MemcpyHostToDevice,
                                            ctx_->gpu_stream()));
      }
      if (!is_device_ptr(fk)) {
        fkBuffer = create_buffer<gpu::ComplexType<T>>(ctx_->allocators().gpu(), N_);
        fkDevice = fkBuffer.get();
      }

      gpu::stream_synchronize(ctx_->gpu_stream()); // cufinufft cannot be asigned a stream
      planGPU_.value().execute(cjDevice, fkDevice);

      if(fkBuffer) {
        gpu::check_status(
            gpu::memcpy_async(fk, fkDevice, N_ * sizeof(gpu::ComplexType<T>),
                              gpu::flag::MemcpyDeviceToHost, nullptr));
      }


      gpu::stream_synchronize(nullptr);
#else
      throw GPUSupportError();
#endif
    }
  }

  int M_, N_;
  std::shared_ptr<ContextInternal> ctx_;
  std::optional<Nufft3d3Host<T>> planHost_;
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
  std::optional<Nufft3d3GPU<T>> planGPU_;
#endif
};

Nufft3d3::Nufft3d3(const Context &ctx, int iflag, double tol, int numTrans,
                   int M, const double *x, const double *y, const double *z,
                   int N, const double *s, const double *t, const double *u)
    : plan_(new Nufft3d3Internal<double>(InternalContextAccessor::get(ctx),
                                         iflag, tol, numTrans, M, x, y, z, N, s,
                                         t, u),
            [](auto &&ptr) {
              delete reinterpret_cast<Nufft3d3Internal<double> *>(ptr);
            }) {}

void Nufft3d3::execute(const std::complex<double> *cj, std::complex<double> *fk) {

  reinterpret_cast<Nufft3d3Internal<double> *>(plan_.get())->exec(cj, fk);
}

Nufft3d3f::Nufft3d3f(const Context &ctx, int iflag, float tol, int numTrans,
                     int M, const float *x, const float *y, const float *z,
                     int N, const float *s, const float *t, const float *u)
    : plan_(new Nufft3d3Internal<float>(InternalContextAccessor::get(ctx),
                                        iflag, tol, numTrans, M, x, y, z, N, s,
                                        t, u),
            [](auto &&ptr) {
              delete reinterpret_cast<Nufft3d3Internal<float> *>(ptr);
            }) {}

void Nufft3d3f::execute(const std::complex<float> *cj, std::complex<float> *fk) {

  reinterpret_cast<Nufft3d3Internal<float> *>(plan_.get())->exec(cj, fk);
}


extern "C" {
BLUEBILD_EXPORT BluebildError bluebild_nufft3d3_create_s(
    BluebildContext ctx, int iflag, float tol, int numTrans, int M,
    const float *x, const float *y, const float *z, int N, const float *s,
    const float *t, const float *u, BluebildNufft3d3* plan) {
  if (!ctx) {
    return BLUEBILD_INVALID_HANDLE_ERROR;
  }
  try {
    *plan = new Nufft3d3f(*reinterpret_cast<Context *>(ctx), iflag, tol,
                         numTrans, M, x, y, z, N, s, t, u);
  } catch (const bluebild::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BLUEBILD_UNKNOWN_ERROR;
  }
  return BLUEBILD_SUCCESS;

}

BLUEBILD_EXPORT BluebildError bluebild_nufft3d3_create_d(
    BluebildContext ctx, int iflag, double tol, int numTrans, int M,
    const double *x, const double *y, const double *z, int N, const double *s,
    const double *t, const double *u, BluebildNufft3d3* plan) {
  if (!ctx) {
    return BLUEBILD_INVALID_HANDLE_ERROR;
  }
  try {
    *plan = new Nufft3d3(*reinterpret_cast<Context *>(ctx), iflag, tol,
                         numTrans, M, x, y, z, N, s, t, u);
  } catch (const bluebild::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BLUEBILD_UNKNOWN_ERROR;
  }
  return BLUEBILD_SUCCESS;

}

BLUEBILD_EXPORT BluebildError bluebild_nufft3d3_destroy_s(BluebildNufft3d3f *plan) {
  if (!plan) {
    return BLUEBILD_INVALID_HANDLE_ERROR;
  }
  try {
    delete reinterpret_cast<Nufft3d3f *>(*plan);
    *plan = nullptr;
  } catch (const bluebild::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BLUEBILD_UNKNOWN_ERROR;
  }
  return BLUEBILD_SUCCESS;
}

BLUEBILD_EXPORT BluebildError bluebild_nufft3d3_destroy_d(BluebildNufft3d3 *plan) {
  if (!plan) {
    return BLUEBILD_INVALID_HANDLE_ERROR;
  }
  try {
    delete reinterpret_cast<Nufft3d3 *>(*plan);
    *plan = nullptr;
  } catch (const bluebild::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BLUEBILD_UNKNOWN_ERROR;
  }
  return BLUEBILD_SUCCESS;
}

BLUEBILD_EXPORT BluebildError
bluebild_nufft3d3_execute_s(BluebildNufft3d3f plan, const void *cj, void *fk) {
  if (!plan) {
    return BLUEBILD_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<Nufft3d3f *>(plan)->execute(
        reinterpret_cast<const std::complex<float> *>(cj),
        reinterpret_cast<std::complex<float> *>(fk));
  } catch (const bluebild::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BLUEBILD_UNKNOWN_ERROR;
  }
  return BLUEBILD_SUCCESS;
}

BLUEBILD_EXPORT BluebildError bluebild_nufft3d3_execute_d(BluebildNufft3d3 plan,
                                                          const void *cj,
                                                          void *fk) {
  if (!plan) {
    return BLUEBILD_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<Nufft3d3 *>(plan)->execute(
        reinterpret_cast<const std::complex<double> *>(cj),
        reinterpret_cast<std::complex<double> *>(fk));
  } catch (const bluebild::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BLUEBILD_UNKNOWN_ERROR;
  }
  return BLUEBILD_SUCCESS;
}

}

} // namespace bluebild
