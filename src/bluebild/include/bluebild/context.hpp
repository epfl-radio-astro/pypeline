#pragma once

#include <memory>

#include "bluebild/config.h"
#include "bluebild/bluebild.h"

namespace bluebild {

class ContextInternal;

struct InternalContextAccessor;

class BLUEBILD_EXPORT Context {
public:
  /**
   * Constructor of Context with default configuration for given processing
   * unit.
   *
   * @param[in] pu Processing unit to be used for computations.
   */
  explicit Context(BluebildProcessingUnit pu);

  /**
   * Default move constructor.
   */
  Context(Context &&) = default;

  /**
   * Disabled copy constructor.
   */
  Context(const Context &) = delete;

  /**
   * Default move assignment operator.
   */
  Context &operator=(Context &&) = default;

  /**
   * Disabled copy assignment operator.
   */
  Context &operator=(const Context &) = delete;

  /**
   * Access a Context parameter.
   * @return Processing unit used.
   */
  auto processing_unit() const -> BluebildProcessingUnit;

private:
  friend InternalContextAccessor;

  std::shared_ptr<ContextInternal> ctx_;
};

} // namespace bluebild
