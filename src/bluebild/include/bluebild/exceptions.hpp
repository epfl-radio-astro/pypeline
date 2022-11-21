#pragma once

#include <stdexcept>

#include "bluebild/config.h"
#include "bluebild/errors.h"

namespace bluebild {

/**
 * A generic error. Base type for all other exceptions.
 */
class BLUEBILD_EXPORT GenericError : public std::exception {
public:
  const char* what() const noexcept override { return "BLUEBILD: Generic error"; }

  virtual BluebildError error_code() const noexcept {
    return BluebildError::BLUEBILD_UNKNOWN_ERROR;
  }
};

class BLUEBILD_EXPORT InternalError : public GenericError {
public:
  const char* what() const noexcept override { return "BLUEBILD: Internal error"; }

  BluebildError error_code() const noexcept override {
    return BluebildError::BLUEBILD_INTERNAL_ERROR;
  }
};

class BLUEBILD_EXPORT InvalidParameterError : public GenericError {
public:
  const char* what() const noexcept override { return "BLUEBILD: Invalid parameter error"; }

  BluebildError error_code() const noexcept override {
    return BluebildError::BLUEBILD_INVALID_PARAMETER_ERROR;
  }
};

class BLUEBILD_EXPORT InvalidPointerError : public GenericError {
public:
  const char* what() const noexcept override { return "BLUEBILD: Invalid pointer error"; }

  BluebildError error_code() const noexcept override {
    return BluebildError::BLUEBILD_INVALID_POINTER_ERROR;
  }
};

class BLUEBILD_EXPORT InvalidAllocatorFunctionError : public GenericError {
public:
  const char* what() const noexcept override {
    return "BLUEBILD: Invalid allocator function error";
  }

  BluebildError error_code() const noexcept override {
    return BluebildError::BLUEBILD_INVALID_ALLOCATOR_FUNCTION;
  }
};

class BLUEBILD_EXPORT EigensolverError : public GenericError {
public:
  const char* what() const noexcept override { return "BLUEBILD: Eigensolver error"; }

  BluebildError error_code() const noexcept override {
    return BluebildError::BLUEBILD_EIGENSOLVER_ERROR;
  }
};

class BLUEBILD_EXPORT FiNUFFTError : public GenericError {
public:
  const char* what() const noexcept override { return "BLUEBILD: fiNUFFT error"; }

  BluebildError error_code() const noexcept override {
    return BluebildError::BLUEBILD_FINUFFT_ERROR;
  }
};

class BLUEBILD_EXPORT NotImplementedError : public GenericError {
public:
  const char* what() const noexcept override { return "BLUEBILD: Not implemented"; }

  BluebildError error_code() const noexcept override {
    return BluebildError::BLUEBILD_NOT_IMPLEMENTED_ERROR;
  }
};

class BLUEBILD_EXPORT GPUError : public GenericError {
public:
  const char* what() const noexcept override { return "BLUEBILD: GPU error"; }

  BluebildError error_code() const noexcept override { return BluebildError::BLUEBILD_GPU_ERROR; }
};

class BLUEBILD_EXPORT GPUSupportError : public GPUError {
public:
  const char* what() const noexcept override { return "BLUEBILD: Not compiled with GPU support"; }

  BluebildError error_code() const noexcept override {
    return BluebildError::BLUEBILD_GPU_SUPPORT_ERROR;
  }
};

class BLUEBILD_EXPORT GPUAllocationError : public GPUError {
public:
  const char* what() const noexcept override { return "BLUEBILD: GPU allocation error"; }

  BluebildError error_code() const noexcept override {
    return BluebildError::BLUEBILD_GPU_ALLOCATION_ERROR;
  }
};

class BLUEBILD_EXPORT GPULaunchError : public GPUError {
public:
  const char* what() const noexcept override { return "BLUEBILD: GPU launch error"; }

  BluebildError error_code() const noexcept override {
    return BluebildError::BLUEBILD_GPU_LAUNCH_ERROR;
  }
};

class BLUEBILD_EXPORT GPUNoDeviceError : public GPUError {
public:
  const char* what() const noexcept override { return "BLUEBILD: GPU no device error"; }

  BluebildError error_code() const noexcept override {
    return BluebildError::BLUEBILD_GPU_NO_DEVICE_ERROR;
  }
};

class BLUEBILD_EXPORT GPUInvalidValueError : public GPUError {
public:
  const char* what() const noexcept override { return "BLUEBILD: GPU invalid value error"; }

  BluebildError error_code() const noexcept override {
    return BluebildError::BLUEBILD_GPU_INVALID_VALUE_ERROR;
  }
};

class BLUEBILD_EXPORT GPUInvalidDevicePointerError : public GPUError {
public:
  const char* what() const noexcept override {
    return "BLUEBILD: GPU invalid device pointer error";
  }

  BluebildError error_code() const noexcept override {
    return BluebildError::BLUEBILD_GPU_INVALID_DEVICE_POINTER_ERROR;
  }
};

class BLUEBILD_EXPORT GPUBlasError : public GPUError {
public:
  const char* what() const noexcept override { return "BLUEBILD: GPU BLAS error"; }

  BluebildError error_code() const noexcept override {
    return BluebildError::BLUEBILD_GPU_BLAS_ERROR;
  }
};

}  // namespace bluebild
