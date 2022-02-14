
#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "context_internal.hpp"

namespace bluebild {

extern "C" {
BLUEBILD_EXPORT BluebildError bluebild_ctx_create(BluebildProcessingUnit pu, BluebildContext* ctx) {
  int nEigOut;
  try {
    *reinterpret_cast<ContextInternal**>(ctx) = new ContextInternal(pu);
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
    delete *reinterpret_cast<ContextInternal**>(ctx);
    *reinterpret_cast<ContextInternal**>(ctx) = nullptr;
  } catch (const bluebild::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BLUEBILD_UNKNOWN_ERROR;
  }
  return BLUEBILD_SUCCESS;
}
}
}  // namespace bluebild
