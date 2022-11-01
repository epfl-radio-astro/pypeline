include(CMakeFindDependencyMacro)
macro(find_dependency_components)
	if(${ARGV0}_FOUND AND ${CMAKE_VERSION} VERSION_LESS "3.15.0")
		# find_dependency does not handle new components correctly before 3.15.0
		set(${ARGV0}_FOUND FALSE)
	endif()
	find_dependency(${ARGV})
endmacro()

# options used for building library
set(BLUEBILD_GPU CUDA)
set(BLUEBILD_CUDA ON)
set(BLUEBILD_ROCM OFF)

