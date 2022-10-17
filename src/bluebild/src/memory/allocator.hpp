#pragma once 

#include <cstddef>

#include "bluebild/config.h"
#include "bluebild/exceptions.hpp"

namespace bluebild {

class Allocator {
public:
  Allocator() = default;

  Allocator(const Allocator&) = delete;

  Allocator(Allocator&&) = default;

  auto operator=(const Allocator&) -> Allocator& = delete;

  auto operator=(Allocator&&) -> Allocator& = default;

  virtual ~Allocator() = default;

  virtual auto allocate(std::size_t size) -> void* = 0;

  virtual auto deallocate(void* ptr) -> void = 0;

  virtual auto size() -> std::uint_least64_t = 0;

    virtual auto print() -> void = 0;
};
}  // namespace bluebild

