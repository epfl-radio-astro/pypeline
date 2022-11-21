#pragma once

#include "bluebild/config.h"

enum BluebildError {
  /**
   * Success. No error.
   */
  BLUEBILD_SUCCESS,
  /**
   * Unknown error.
   */
  BLUEBILD_UNKNOWN_ERROR,
  /**
   * Internal error.
   */
  BLUEBILD_INTERNAL_ERROR,
  /**
   * Invalid parameter error.
   */
  BLUEBILD_INVALID_PARAMETER_ERROR,
  /**
   * Invalid pointer error.
   */
  BLUEBILD_INVALID_POINTER_ERROR,
  /**
   * Invalid handle error.
   */
  BLUEBILD_INVALID_HANDLE_ERROR,
  /**
   * Eigensolver error.
   */
  BLUEBILD_EIGENSOLVER_ERROR,
  /**
   * fiNUFFT error.
   */
  BLUEBILD_FINUFFT_ERROR,
  /**
   * Not Implemented error.
   */
  BLUEBILD_NOT_IMPLEMENTED_ERROR,
  /**
   * GPU error.
   */
  BLUEBILD_GPU_ERROR,
  /**
   * GPU support error.
   */
  BLUEBILD_GPU_SUPPORT_ERROR,
  /**
   * GPU allocation error.
   */
  BLUEBILD_GPU_ALLOCATION_ERROR,
  /**
   * GPU launch error.
   */
  BLUEBILD_GPU_LAUNCH_ERROR,
  /**
   * GPU no device error.
   */
  BLUEBILD_GPU_NO_DEVICE_ERROR,
  /**
   * GPU invalid value error.
   */
  BLUEBILD_GPU_INVALID_VALUE_ERROR,
  /**
   * Invalid device pointer error.
   */
  BLUEBILD_GPU_INVALID_DEVICE_POINTER_ERROR,
  /**
   * GPU blas error.
   */
  BLUEBILD_GPU_BLAS_ERROR,
  /**
   * Invalid allocator function error.
   */
  BLUEBILD_INVALID_ALLOCATOR_FUNCTION
};

#ifndef __cplusplus
/*! \cond PRIVATE */
// C only
typedef enum BluebildError BluebildError;
/*! \endcond */
#endif  // cpp
