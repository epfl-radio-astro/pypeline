#pragma once

#include "bluebild/config.h"

#if defined(BLUEBILD_CUDA)
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#define GPU_PREFIX(val) cuda##val

#elif defined(BLUEBILD_ROCM)
#include <hip/hip_runtime_api.h>
#define GPU_PREFIX(val) hip##val
#endif

// only declare namespace members if GPU support is enabled
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)

#include <utility>
#include <complex>

#include "bluebild/exceptions.hpp"

namespace bluebild {
namespace gpu {

using StatusType = GPU_PREFIX(Error_t);
using StreamType = GPU_PREFIX(Stream_t);
using EventType = GPU_PREFIX(Event_t);

#ifdef BLUEBILD_CUDA
using PointerAttributes = GPU_PREFIX(PointerAttributes);

using ComplexDoubleType = cuDoubleComplex;
using ComplexFloatType = cuComplex;
#else
using PointerAttributes = GPU_PREFIX(PointerAttribute_t);
using ComplexDoubleType = hipDoubleComplex;
using ComplexFloatType = hipComplex;
#endif

template <typename T>
using ComplexType = std::conditional_t<std::is_same<T, float>{}, ComplexFloatType,
                                       std::conditional_t<std::is_same<T, std::complex<float>>{},
                                                          ComplexFloatType, ComplexDoubleType>>;

namespace status {
// error / return values
constexpr StatusType Success = GPU_PREFIX(Success);
constexpr StatusType ErrorMemoryAllocation = GPU_PREFIX(ErrorMemoryAllocation);
constexpr StatusType ErrorLaunchOutOfResources = GPU_PREFIX(ErrorLaunchOutOfResources);
constexpr StatusType ErrorInvalidValue = GPU_PREFIX(ErrorInvalidValue);
constexpr StatusType ErrorInvalidResourceHandle = GPU_PREFIX(ErrorInvalidResourceHandle);
constexpr StatusType ErrorInvalidDevice = GPU_PREFIX(ErrorInvalidDevice);
constexpr StatusType ErrorInvalidMemcpyDirection = GPU_PREFIX(ErrorInvalidMemcpyDirection);
constexpr StatusType ErrorInvalidDevicePointer = GPU_PREFIX(ErrorInvalidDevicePointer);
constexpr StatusType ErrorInitializationError = GPU_PREFIX(ErrorInitializationError);
constexpr StatusType ErrorNoDevice = GPU_PREFIX(ErrorNoDevice);
constexpr StatusType ErrorNotReady = GPU_PREFIX(ErrorNotReady);
constexpr StatusType ErrorUnknown = GPU_PREFIX(ErrorUnknown);
constexpr StatusType ErrorPeerAccessNotEnabled = GPU_PREFIX(ErrorPeerAccessNotEnabled);
constexpr StatusType ErrorPeerAccessAlreadyEnabled = GPU_PREFIX(ErrorPeerAccessAlreadyEnabled);
constexpr StatusType ErrorHostMemoryAlreadyRegistered =
    GPU_PREFIX(ErrorHostMemoryAlreadyRegistered);
constexpr StatusType ErrorHostMemoryNotRegistered = GPU_PREFIX(ErrorHostMemoryNotRegistered);
constexpr StatusType ErrorUnsupportedLimit = GPU_PREFIX(ErrorUnsupportedLimit);
}  // namespace status

// flags to pass to GPU API
namespace flag {
constexpr auto HostRegisterDefault = GPU_PREFIX(HostRegisterDefault);
constexpr auto HostRegisterPortable = GPU_PREFIX(HostRegisterPortable);
constexpr auto HostRegisterMapped = GPU_PREFIX(HostRegisterMapped);
constexpr auto HostRegisterIoMemory = GPU_PREFIX(HostRegisterIoMemory);

constexpr auto StreamDefault = GPU_PREFIX(StreamDefault);
constexpr auto StreamNonBlocking = GPU_PREFIX(StreamNonBlocking);

constexpr auto MemoryTypeHost = GPU_PREFIX(MemoryTypeHost);
constexpr auto MemoryTypeDevice = GPU_PREFIX(MemoryTypeDevice);
#if (CUDART_VERSION >= 10000)
constexpr auto MemoryTypeUnregistered = GPU_PREFIX(MemoryTypeUnregistered);
constexpr auto MemoryTypeManaged = GPU_PREFIX(MemoryTypeManaged);
#endif

constexpr auto MemcpyDefault = GPU_PREFIX(MemcpyDefault);
constexpr auto MemcpyHostToDevice = GPU_PREFIX(MemcpyHostToDevice);
constexpr auto MemcpyDeviceToHost = GPU_PREFIX(MemcpyDeviceToHost);
constexpr auto MemcpyDeviceToDevice = GPU_PREFIX(MemcpyDeviceToDevice);

constexpr auto EventDefault = GPU_PREFIX(EventDefault);
constexpr auto EventBlockingSync = GPU_PREFIX(EventBlockingSync);
constexpr auto EventDisableTiming = GPU_PREFIX(EventDisableTiming);
constexpr auto EventInterprocess = GPU_PREFIX(EventInterprocess);
}  // namespace flag

// ==================================
// Error check functions
// ==================================
inline auto check_status(StatusType error) -> void {
  if (error != status::Success) {
    if (error == status::ErrorMemoryAllocation) throw GPUAllocationError();
    if (error == status::ErrorLaunchOutOfResources) throw GPULaunchError();
    if (error == status::ErrorNoDevice) throw GPUNoDeviceError();
    if (error == status::ErrorInvalidValue) throw GPUInvalidValueError();
    if (error == status::ErrorInvalidDevicePointer) throw GPUInvalidDevicePointerError();

    throw GPUError();
  }
}

// ==================================
// Forwarding functions of to GPU API
// ==================================
template <typename... ARGS>
inline auto host_register(ARGS&&... args) -> StatusType {
  return GPU_PREFIX(HostRegister)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto host_unregister(ARGS&&... args) -> StatusType {
  return GPU_PREFIX(HostUnregister)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto stream_create_with_flags(ARGS&&... args) -> StatusType {
  return GPU_PREFIX(StreamCreateWithFlags)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto stream_destroy(ARGS&&... args) -> StatusType {
  return GPU_PREFIX(StreamDestroy)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto stream_wait_event(ARGS&&... args) -> StatusType {
  return GPU_PREFIX(StreamWaitEvent)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto event_create_with_flags(ARGS&&... args) -> StatusType {
  return GPU_PREFIX(EventCreateWithFlags)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto event_destroy(ARGS&&... args) -> StatusType {
  return GPU_PREFIX(EventDestroy)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto event_record(ARGS&&... args) -> StatusType {
  return GPU_PREFIX(EventRecord)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto event_synchronize(ARGS&&... args) -> StatusType {
  return GPU_PREFIX(EventSynchronize)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto malloc(ARGS&&... args) -> StatusType {
  return GPU_PREFIX(Malloc)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto malloc_host(ARGS&&... args) -> StatusType {
#if defined(BLUEBILD_CUDA)
  return GPU_PREFIX(MallocHost)(std::forward<ARGS>(args)...);
#else
  // hip deprecated hipMallocHost in favour of hipHostMalloc
  return GPU_PREFIX(HostMalloc)(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
inline auto free(ARGS&&... args) -> StatusType {
  return GPU_PREFIX(Free)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto free_host(ARGS&&... args) -> StatusType {
#if defined(BLUEBILD_CUDA)
  return GPU_PREFIX(FreeHost)(std::forward<ARGS>(args)...);
#else
  // hip deprecated hipFreeHost in favour of hipHostFree
  return GPU_PREFIX(HostFree)(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
inline auto memcpy(ARGS&&... args) -> StatusType {
  return GPU_PREFIX(Memcpy)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto memcpy_2d(ARGS&&... args) -> StatusType {
  return GPU_PREFIX(Memcpy2D)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto memcpy_async(ARGS&&... args) -> StatusType {
  return GPU_PREFIX(MemcpyAsync)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto memcpy_2d_async(ARGS&&... args) -> StatusType {
  return GPU_PREFIX(Memcpy2DAsync)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto get_device(ARGS&&... args) -> StatusType {
  return GPU_PREFIX(GetDevice)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto set_device(ARGS&&... args) -> StatusType {
  return GPU_PREFIX(SetDevice)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto get_device_count(ARGS&&... args) -> StatusType {
  return GPU_PREFIX(GetDeviceCount)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto stream_synchronize(ARGS&&... args) -> StatusType {
  return GPU_PREFIX(StreamSynchronize)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto memset_async(ARGS&&... args) -> StatusType {
  return GPU_PREFIX(MemsetAsync)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto memset_2d_async(ARGS&&... args) -> StatusType {
  return GPU_PREFIX(Memset2DAsync)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto pointer_get_attributes(ARGS&&... args) -> StatusType {
  return GPU_PREFIX(PointerGetAttributes)(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto mem_get_info(ARGS&&... args) -> StatusType {
  return GPU_PREFIX(MemGetInfo)(std::forward<ARGS>(args)...);
}

inline auto get_last_error() -> StatusType { return GPU_PREFIX(GetLastError)(); }

inline auto device_synchronize() -> StatusType { return GPU_PREFIX(DeviceSynchronize)(); }

}  // namespace gpu
}  // namespace bluebild

#undef GPU_PREFIX

#endif  // defined BLUEBILD_CUDA || BLUEBILD_ROCM
