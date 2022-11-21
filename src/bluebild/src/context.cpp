
#include "bluebild/config.h"
#include "context_internal.hpp"
#include "bluebild/context.hpp"

namespace bluebild {

Context::Context(BluebildProcessingUnit pu) : ctx_(new ContextInternal(pu)) {}

BluebildProcessingUnit Context::processing_unit() const {
  return ctx_->processing_unit();
}

extern "C" {
BLUEBILD_EXPORT BluebildError bluebild_ctx_create(BluebildProcessingUnit pu, BluebildContext* ctx) {
  int nEigOut;
  try {
    *reinterpret_cast<Context**>(ctx) = new Context(pu);
  } catch (const bluebild::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BLUEBILD_UNKNOWN_ERROR;
  }
  return BLUEBILD_SUCCESS;
}

BLUEBILD_EXPORT BluebildError bluebild_ctx_destroy(BluebildContext* ctx) {
  if (!ctx || !(*ctx)) {
    return BLUEBILD_INVALID_HANDLE_ERROR;
  }
  try {
    delete *reinterpret_cast<Context**>(ctx);
    *reinterpret_cast<Context**>(ctx) = nullptr;
  } catch (const bluebild::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BLUEBILD_UNKNOWN_ERROR;
  }
  return BLUEBILD_SUCCESS;
}
}
}  // namespace bluebild
