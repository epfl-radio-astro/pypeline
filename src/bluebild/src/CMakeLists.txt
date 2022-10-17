set(BLUEBILD_SOURCE_FILES
	eigensolver.cpp
	gram_matrix.cpp
	intensity_field_data.cpp
	sensitivity_field_data.cpp
	virtual_vis.cpp
	nufft_3d3.cpp
    standard_synthesizer.cpp
	context.cpp
	periodic_synthesis.cpp
	host/eigensolver_host.cpp
	host/gram_matrix_host.cpp
	host/sensitivity_field_data_host.cpp
	host/intensity_field_data_host.cpp
	host/periodic_synthesis_host.cpp
	host/nufft_3d3_host.cpp
	host/virtual_vis_host.cpp
	host/ss_host.cpp
	host/gemmexp.cpp
	host/standard_synthesizer_host.cpp
	)

if(BLUEBILD_CUDA OR BLUEBILD_ROCM)
	list(APPEND BLUEBILD_SOURCE_FILES
		gpu/eigensolver_gpu.cpp
		gpu/sensitivity_field_data_gpu.cpp
		gpu/intensity_field_data_gpu.cpp
		gpu/gram_matrix_gpu.cpp
		gpu/nufft_3d3_gpu.cpp
		gpu/periodic_synthesis_gpu.cpp
		gpu/virtual_vis_gpu.cpp
		gpu/standard_synthesizer_gpu.cpp
		gpu/kernels/reverse.cu
		gpu/kernels/inv_square.cu
		gpu/kernels/min_diff.cu
		gpu/kernels/gram.cu
		gpu/kernels/scale_matrix.cu
		gpu/kernels/apply_filter.cu
		gpu/kernels/add_vector.cu
		gpu/kernels/standard_synthesizer.cu
		)
endif()


# Creates bluebild library with given name. All common target modifications should be done here.
macro(bluebild_create_library _TARGET_NAME)
	add_library(${_TARGET_NAME} ${BLUEBILD_LIBRARY_TYPE}
		${BLUEBILD_SOURCE_FILES}
		)

	set_property(TARGET ${_TARGET_NAME} PROPERTY VERSION ${BLUEBILD_VERSION})
	set_property(TARGET ${_TARGET_NAME} PROPERTY SOVERSION ${BLUEBILD_SO_VERSION})


	# Don't export any symbols of external static libaries. Only works on linux.
	if(UNIX AND NOT APPLE)
		if(${CMAKE_VERSION} VERSION_LESS "3.13.5") 
			target_link_libraries(${_TARGET_NAME} PRIVATE "-Wl,--exclude-libs,ALL")
		else()
			target_link_options(${_TARGET_NAME} PRIVATE "-Wl,--exclude-libs,ALL")
		endif()
	endif()

	# All .cu files are self-contained. Device linking can have issues with propageted linker flags of other targets like MPI.
	if(BLUEBILD_CUDA)
		set_property(TARGET ${_TARGET_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
		set_property(TARGET ${_TARGET_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
	endif()

	target_compile_options(${_TARGET_NAME} PRIVATE ${BLUEBILD_DEFINITIONS} ${BLUEBILD_EXTERNAL_COMPILE_OPTIONS})
	target_include_directories(${_TARGET_NAME} PRIVATE ${BLUEBILD_EXTERNAL_INCLUDE_DIRS} ${BLUEBILD_INCLUDE_DIRS})


	target_link_libraries(${_TARGET_NAME} PRIVATE ${BLUEBILD_EXTERNAL_LIBS})

	target_include_directories(${_TARGET_NAME} INTERFACE ${BLUEBILD_INTERFACE_INCLUDE_DIRS})
	target_include_directories(${_TARGET_NAME} INTERFACE $<INSTALL_INTERFACE:include>) # for install(EXPORT ...)
	target_include_directories(${_TARGET_NAME} INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>) # for export(...)

endmacro()

bluebild_create_library(bluebild)
set_target_properties(bluebild PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE CXX_VISIBILITY_PRESET hidden)

if(BLUEBILD_BUILD_TESTS)
	# create library with default visibility if tests are build, to allow linking to internal symbols
	bluebild_create_library(bluebild_test)
	set_target_properties(bluebild_test PROPERTIES VISIBILITY_INLINES_HIDDEN FALSE CXX_VISIBILITY_PRESET default)
	target_compile_options(bluebild_test PUBLIC -DBLUEBILD_STATIC_DEFINE)
	# enable internal timings
	target_compile_options(bluebild_test PUBLIC -DBLUEBILD_TIMING)
endif()

# generate export header to control symbol visibility
include(GenerateExportHeader)
generate_export_header(bluebild)
configure_file("${CMAKE_CURRENT_BINARY_DIR}/bluebild_export.h"
	"${PROJECT_BINARY_DIR}/bluebild/bluebild_export.h"
	COPYONLY
)

# set packge config names
get_target_property(_LIB_TYPE bluebild TYPE)
if(_LIB_TYPE STREQUAL "STATIC_LIBRARY")
	set(BLUEBILD_VERSION_FILE "BLUEBILDStaticConfigVersion.cmake")
	set(BLUEBILD_CONFIG_FILE "BLUEBILDStaticConfig.cmake")
	set(BLUEBILD_TARGETS_FILE "BLUEBILDStaticTargets.cmake")
else()
	set(BLUEBILD_VERSION_FILE "BLUEBILDSharedConfigVersion.cmake")
	set(BLUEBILD_CONFIG_FILE "BLUEBILDSharedConfig.cmake")
	set(BLUEBILD_TARGETS_FILE "BLUEBILDSharedTargets.cmake")
endif()


# generate cmake package
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
	"${PROJECT_BINARY_DIR}/${BLUEBILD_VERSION_FILE}"
	VERSION ${Upstream_VERSION}
	COMPATIBILITY AnyNewerVersion
)
export(TARGETS bluebild NAMESPACE BLUEBILD:: FILE ${PROJECT_BINARY_DIR}/${BLUEBILD_TARGETS_FILE})
configure_file(${PROJECT_SOURCE_DIR}/cmake/${BLUEBILD_CONFIG_FILE}
	"${PROJECT_BINARY_DIR}/${BLUEBILD_CONFIG_FILE}"
	@ONLY
)
configure_file(${PROJECT_SOURCE_DIR}/cmake/BLUEBILDConfig.cmake
	"${PROJECT_BINARY_DIR}/BLUEBILDConfig.cmake"
	COPYONLY
)
configure_file(${PROJECT_SOURCE_DIR}/cmake/BLUEBILDConfigVersion.cmake
	"${PROJECT_BINARY_DIR}/BLUEBILDConfigVersion.cmake"
	COPYONLY
)
configure_file(${PROJECT_SOURCE_DIR}/cmake/BLUEBILDTargets.cmake
	"${PROJECT_BINARY_DIR}/BLUEBILDTargets.cmake"
	COPYONLY
)

# installation commands
if(BLUEBILD_INSTALL STREQUAL "LIB")
	install(TARGETS bluebild DESTINATION ${CMAKE_INSTALL_LIBDIR} EXPORT BLUEBILDTargets)
	install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/bluebild DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.f90")
	install(FILES ${PROJECT_BINARY_DIR}/bluebild/config.h "${PROJECT_BINARY_DIR}/bluebild/bluebild_export.h" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/bluebild)
	install(EXPORT BLUEBILDTargets NAMESPACE BLUEBILD:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BLUEBILD FILE ${BLUEBILD_TARGETS_FILE})
	install(
	  FILES
		"${PROJECT_BINARY_DIR}/BLUEBILDConfig.cmake"
		"${PROJECT_BINARY_DIR}/BLUEBILDTargets.cmake"
		"${PROJECT_BINARY_DIR}/BLUEBILDConfigVersion.cmake"
		"${PROJECT_BINARY_DIR}/${BLUEBILD_CONFIG_FILE}"
		"${PROJECT_BINARY_DIR}/${BLUEBILD_VERSION_FILE}"
	  DESTINATION
	    ${CMAKE_INSTALL_LIBDIR}/cmake/BLUEBILD
	)

	install(DIRECTORY "${PROJECT_SOURCE_DIR}/cmake/modules"
		DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/BLUEBILD"
        FILES_MATCHING PATTERN "*.cmake")

endif()
