#pragma once

#include <memory>
#include <functional>
#include "bluebild/config.h"

#include "memory/allocator.hpp"

namespace bluebild {

template <typename T>
using BufferType = std::unique_ptr<T[], std::function<void(T*)>>;

// Create buffer by allocing memory through an allocator
template <typename T>
auto create_buffer(const std::shared_ptr<Allocator>& alloc, std::size_t size) -> BufferType<T> {
  return {static_cast<T*>(size ? alloc->allocate(size * sizeof(T)) : nullptr), [=](T* ptr) {
            if (ptr) alloc->deallocate(ptr);
          }};
}

}  // namespace bluebild
