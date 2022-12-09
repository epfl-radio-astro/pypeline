#pragma once

#include <memory>
#include <cassert>
#include <functional>
#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "bluebild/exceptions.hpp"
#include "bluebild/context.hpp"
#include "memory/allocator_collection.hpp"

#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
#include <cusolverDn.h>
#include "gpu/util/gpu_runtime_api.hpp"
#include "gpu/util/gpu_blas_api.hpp"
#endif

namespace bluebild {


class ContextInternal {
  public:
    explicit ContextInternal(BluebildProcessingUnit pu) {
      if(pu == BLUEBILD_PU_AUTO) {
        // select GPU if available
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
        int deviceId = 0;
        if (gpu::get_device(&deviceId) == gpu::status::Success) {
          pu_ = BLUEBILD_PU_GPU;
        } else {
          pu_ = BLUEBILD_PU_CPU;
        }
#else
        pu_ = BLUEBILD_PU_CPU;
#endif
      } else {
        pu_ = pu;
      }

      if (pu_ == BLUEBILD_PU_GPU) {
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
        // create stream
        gpu::StreamType stream;
        gpu::check_status(gpu::stream_create_with_flags(&stream, gpu::flag::StreamNonBlocking));
        gpuStream_ = std::unique_ptr<gpu::StreamType, std::function<void(gpu::StreamType*)>>(
            new gpu::StreamType(stream), [](gpu::StreamType* ptr) {
              gpu::stream_destroy(*ptr);
              delete ptr;
            });
        // create stream
        gpu::StreamType stream2;
        gpu::check_status(gpu::stream_create_with_flags(&stream2, gpu::flag::StreamNonBlocking));
        gpuStream2_ = std::unique_ptr<gpu::StreamType, std::function<void(gpu::StreamType*)>>(
            new gpu::StreamType(stream2), [](gpu::StreamType* ptr) {
              gpu::stream_destroy(*ptr);
              delete ptr;
            });

        // create blas handle
        gpu::blas::HandleType blasHandle;
        gpu::blas::check_status(gpu::blas::create(&blasHandle));
        gpuBlasHandle_ =
            std::unique_ptr<gpu::blas::HandleType, std::function<void(gpu::blas::HandleType*)>>(
                new gpu::blas::HandleType(blasHandle), [](gpu::blas::HandleType* ptr) {
                  gpu::blas::destroy(*ptr);
                  delete ptr;
                });
        gpu::blas::set_stream(*gpuBlasHandle_, *gpuStream_);

        // create gpu solver
        cusolverDnHandle_t solverHandle;
        if (cusolverDnCreate(&solverHandle) != CUSOLVER_STATUS_SUCCESS) throw GPUError();
        gpuSolverHandle_ =
            std::unique_ptr<cusolverDnHandle_t, std::function<void(cusolverDnHandle_t*)>>(
                new cusolverDnHandle_t(solverHandle), [](cusolverDnHandle_t* ptr) {
                  cusolverDnDestroy(*ptr);
                  delete ptr;
                });

        if (cusolverDnSetStream(solverHandle, *gpuStream_) != CUSOLVER_STATUS_SUCCESS)
          throw GPUError();

#else
        throw GPUSupportError();  // CPU not implemented yet
#endif
      }
    }

#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
    auto gpu_stream() const -> const gpu::StreamType& {
      return *gpuStream_;
    }
    auto gpu_stream2() const -> const gpu::StreamType& {
      return *gpuStream2_;
    }

    auto gpu_blas_handle() const -> const gpu::blas::HandleType& {
      return *gpuBlasHandle_;
    }

    auto gpu_solver_handle() const -> const cusolverDnHandle_t& {
      return *gpuSolverHandle_;
    }
#endif

    auto allocators() -> AllocatorCollection& { return allocators_; }

    auto processing_unit() const -> BluebildProcessingUnit { return pu_; }

  private:
    BluebildProcessingUnit pu_;
    AllocatorCollection allocators_;

#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
    std::unique_ptr<gpu::StreamType, std::function<void(gpu::StreamType*)>> gpuStream_, gpuStream2_;
    std::unique_ptr<gpu::blas::HandleType, std::function<void(gpu::blas::HandleType*)>> gpuBlasHandle_;
    std::unique_ptr<cusolverDnHandle_t, std::function<void(cusolverDnHandle_t*)>> gpuSolverHandle_;
#endif
};


struct InternalContextAccessor {
  static auto get(const Context& ctx) -> const std::shared_ptr<ContextInternal>& {
    return ctx.ctx_;
  }
};

}  // namespace bluebild

