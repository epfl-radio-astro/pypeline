#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <memory>

#include "memory/allocator.hpp"
#include "memory/pool_allocator.hpp"
#include "bluebild/config.h"
#include "bluebild/exceptions.hpp"

#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
#include "gpu/util/gpu_runtime_api.hpp"
#endif

#ifdef BLUEBILD_UMPIRE
#include "memory/umpire_allocator.hpp"
#endif

namespace bluebild {
class AllocatorCollection {
public:
  AllocatorCollection()
      :
#ifdef BLUEBILD_UMPIRE
        allocHost_(new UmpireAllocator("HOST"))
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
        ,
        allocPinned_(new UmpireAllocator("PINNED")),
        allocGPU_(new UmpireAllocator("DEVICE"))
#endif // CUDA / ROCM
#else // UMPIRE
        allocHost_(new PoolAllocator(std::malloc, std::free))
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
        ,
        allocPinned_(new PoolAllocator(
            [](std::size_t size) -> void * {
              void* ptr = nullptr;
              gpu::check_status(gpu::malloc_host(&ptr, size));
              return ptr;
            },
            [](void *ptr) -> void { gpu::free_host(ptr); })),
        allocGPU_(new PoolAllocator(
            [](std::size_t size) -> void * {
              void* ptr = nullptr;
              gpu::check_status(gpu::malloc(&ptr, size));
              return ptr;
            },
            [](void *ptr) -> void { gpu::free(ptr); }))
#endif // CUDA / ROCM
#endif // UMPIRE
  {
  }

  auto host() const -> const std::shared_ptr<Allocator>& { return allocHost_; }

  auto set_host(std::shared_ptr<Allocator> allocator) -> void {
    assert(allocator);
    allocHost_ = std::move(allocator);
  }

#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
  auto pinned() const -> const std::shared_ptr<Allocator>& { return allocPinned_; }

  auto set_pinned(std::shared_ptr<Allocator> allocator) -> void {
    assert(allocator);
    allocPinned_ = std::move(allocator);
  }

  auto gpu() const -> const std::shared_ptr<Allocator>& { return allocGPU_; }

  auto set_gpu(std::shared_ptr<Allocator> allocator) -> void {
    assert(allocator);
    allocGPU_ = std::move(allocator);
  }
#endif

private:
    std::shared_ptr<Allocator> allocHost_;
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
    std::shared_ptr<Allocator> allocPinned_;
    std::shared_ptr<Allocator> allocGPU_;
#endif

};
}  // namespace bluebild
