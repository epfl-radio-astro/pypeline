#pragma once

#include "bluebild/config.h"
#include "memory/allocator.hpp"
#include "bluebild/exceptions.hpp"

#ifdef BLUEBILD_UMPIRE
#include <umpire/ResourceManager.hpp>
#include <umpire/strategy/DynamicPoolList.hpp>


namespace bluebild {

class UmpireAllocator : public Allocator {
public:
  UmpireAllocator(const std::string &location) {
    const auto name = location + "_dynamic_bluebild";
      auto& rm = umpire::ResourceManager::getInstance();
      // Any allocator is stored in global instance. Reuse, if it already exists.
      if (rm.isAllocator(name))
        alloc_ = rm.getAllocator(name);
      else
        alloc_ = rm.makeAllocator<umpire::strategy::DynamicPoolList>(
            name, rm.getAllocator(location), 4096);
  }

  UmpireAllocator(const UmpireAllocator&) = delete;

  UmpireAllocator(UmpireAllocator&&) = default;

  auto operator=(const UmpireAllocator&) -> UmpireAllocator& = delete;

  auto operator=(UmpireAllocator&&) -> UmpireAllocator& = default;

  auto allocate(std::size_t size) -> void* override {
    return alloc_.allocate(size);
  }

  auto deallocate(void* ptr) -> void override {
    alloc_.deallocate(ptr);
  }

  auto size() -> std::uint_least64_t override {
    return alloc_.getActualSize();
  }

  auto print() -> void override {}

private:
  // umpire::ResourceManager manager_;
  umpire::Allocator alloc_;

};
}  // namespace bluebild
#endif
