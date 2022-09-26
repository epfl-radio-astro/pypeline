cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(bluebild LANGUAGES CXX VERSION 0.1.0)
set(BLUEBILD_SO_VERSION 1)
set(BLUEBILD_VERSION ${PROJECT_VERSION})

# allow {module}_ROOT variables to be set
if(POLICY CMP0074)
	cmake_policy(SET CMP0074 NEW)
endif()

# use INTERFACE_LINK_LIBRARIES property if available
if(POLICY CMP0022)
	cmake_policy(SET CMP0022 NEW)
endif()

# set default build type to RELEASE
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
		"Debug" "Release" "MinSizeRel" "RelWithDebInfo"
		)
endif()

# set language and standard
set(CMAKE_CXX_STANDARD 17)

# Get GNU standard install prefixes
include(GNUInstallDirs)

#add local module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake/modules)

# Options
option(BLUEBILD_BUILD_TESTS "Build tests" OFF)
option(BLUEBILD_INSTALL "Enable CMake install commands" ON)

set(BLUEBILD_GPU "OFF" CACHE STRING "GPU backend")
set_property(CACHE BLUEBILD_GPU PROPERTY STRINGS
	"OFF" "CUDA" "ROCM"
	)
set(BUILD_SHARED_LIBS "ON" CACHE STRING "Build as shared library") # default to shared

# Options combination check
set(BLUEBILD_CUDA OFF)
set(BLUEBILD_ROCM OFF)
if(BLUEBILD_GPU)
	if(BLUEBILD_GPU STREQUAL "CUDA")
		set(BLUEBILD_CUDA ON)
	elseif(BLUEBILD_GPU STREQUAL "ROCM")
		set(BLUEBILD_ROCM ON)
	else()
		message(FATAL_ERROR "Invalid GPU backend")
	endif()
endif()

# CUDA
if(BLUEBILD_CUDA)
	enable_language(CUDA)
	if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.17.0") 
		find_package(CUDAToolkit REQUIRED)
	else()
		find_library(CUDA_CUDART_LIBRARY cudart PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
		if(NOT TARGET CUDA::cudart)
			add_library(CUDA::cudart INTERFACE IMPORTED)
		endif()
		set_property(TARGET CUDA::cudart PROPERTY INTERFACE_LINK_LIBRARIES ${CUDA_CUDART_LIBRARY})
		set_property(TARGET CUDA::cudart PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

		find_library(CUDA_CUBLAS_LIBRARY cublas PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
		if(NOT TARGET CUDA::cublas)
			add_library(CUDA::cublas INTERFACE IMPORTED)
		endif()
		set_property(TARGET CUDA::cublas PROPERTY INTERFACE_LINK_LIBRARIES ${CUDA_CUBLAS_LIBRARY})
		set_property(TARGET CUDA::cublas PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

		find_library(CUDA_CUSOLVER_LIBRARY cusolver PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
		if(NOT TARGET CUDA::cusolver)
			add_library(CUDA::cusolver INTERFACE IMPORTED)
		endif()
		set_property(TARGET CUDA::cusolver PROPERTY INTERFACE_LINK_LIBRARIES ${CUDA_CUSOLVER_LIBRARY})
		set_property(TARGET CUDA::cusolver PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
	endif()

	list(APPEND BLUEBILD_EXTERNAL_LIBS CUDA::cudart CUDA::cublas CUDA::cusolver)
endif()

# ROCm
if(BLUEBILD_ROCM)
	message(FATAL_ERROR "ROCM support not yet fully implemented") # TODO: remove when implemented
	find_package(hip CONFIG REQUIRED)
	find_package(rocblas CONFIG REQUIRED)
  list(APPEND BLUEBILD_EXTERNAL_LIBS hip::host roc::rocblas)
endif()

set(BLA_SIZEOF_INTEGER 4) # 32 bit interface to blas / lapack
# BLAS
find_package(BLAS REQUIRED)
if(NOT TARGET BLAS::BLAS)
	# target is only available with CMake 3.18.0 and later
	add_library(BLAS::BLAS INTERFACE IMPORTED)
	set_property(TARGET BLAS::BLAS PROPERTY INTERFACE_LINK_LIBRARIES ${BLAS_LIBRARIES} ${BLAS_LINKER_FLAGS})
endif()
list(APPEND BLUEBILD_EXTERNAL_LIBS BLAS::BLAS)

# LAPACK
find_package(LAPACK REQUIRED)
if(NOT TARGET LAPACK::LAPACK)
	# target is only available with CMake 3.18.0 and later
	add_library(LAPACK::LAPACK INTERFACE IMPORTED)
	set_property(TARGET LAPACK::LAPACK PROPERTY INTERFACE_LINK_LIBRARIES ${LAPACK_LINKER_FLAGS} ${LAPACK_LIBRARIES})
endif()
list(APPEND BLUEBILD_EXTERNAL_LIBS LAPACK::LAPACK)

# check if C api is available for blas and lapack
# include(CheckCXXSymbolExists)
set(CMAKE_REQUIRED_LIBRARIES ${BLUEBILD_EXTERNAL_LIBS})
include(CheckFunctionExists)

unset(BLUEBILD_BLAS_C CACHE) # Result is cached, so change of library will not lead to a new check automatically
CHECK_FUNCTION_EXISTS(cblas_zgemm BLUEBILD_BLAS_C)

unset(BLUEBILD_LAPACK_C CACHE) # Result is cached, so change of library will not lead to a new check automatically
CHECK_FUNCTION_EXISTS(LAPACKE_zheevx BLUEBILD_LAPACK_C)

# generate config.h
configure_file(include/bluebild/config.h.in ${PROJECT_BINARY_DIR}/bluebild/config.h)

list(APPEND BLUEBILD_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src)
list(APPEND BLUEBILD_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)
list(APPEND BLUEBILD_INCLUDE_DIRS ${PROJECT_BINARY_DIR})

#############################################################################
# All include dirs and definitions must be set before sub-directory is added!
#############################################################################
add_subdirectory(src)

# add tests for developement
if(BLUEBILD_BUILD_TESTS)
	add_subdirectory(tests)
endif()
