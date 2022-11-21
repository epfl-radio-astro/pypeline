set(Python_FIND_FRAMEWORK LAST) # Prefer Brew/Conda to Apple framework python
find_package(Python 3.6 COMPONENTS Interpreter Development REQUIRED)

if (BLUEBILD_PYBIND11_DOWNLOAD)
	include(FetchContent)
	FetchContent_Declare(
		pybind11
		GIT_REPOSITORY https://github.com/pybind/pybind11.git
		GIT_TAG        v2.9.2
	)
	FetchContent_MakeAvailable(pybind11)
else()
	# Check if available as python package
	if (Python_EXECUTABLE)
		execute_process(
			COMMAND "${Python_EXECUTABLE}" -c
							"import pybind11; print(pybind11.get_cmake_dir())"
			OUTPUT_VARIABLE _pybind_dir
			OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
	endif()

	find_package(pybind11 CONFIG REQUIRED HINTS ${_pybind_dir})
endif()


pybind11_add_module(pybluebild pybluebild.cpp)
target_link_libraries(pybluebild PRIVATE bluebild)
set_target_properties(pybluebild PROPERTIES
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bluebild"
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bluebild"
	)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/bluebild/__init__.py"
	"${CMAKE_CURRENT_BINARY_DIR}/bluebild/__init__.py"
	COPYONLY
)

if(BLUEBILD_INSTALL STREQUAL "PYTHON")
	# make sure the module can find the library
	if(APPLE)
		set(BLUEBILD_PYTHON_RPATH "@loader_path/_libs" CACHE STRING "RPATH used for python module")
	else()
		set(BLUEBILD_PYTHON_RPATH "$ORIGIN/_libs" CACHE STRING "RPATH used for python module")
	endif()
	if(BLUEBILD_PYTHON_RPATH)
		set_target_properties(pybluebild PROPERTIES
				INSTALL_RPATH_USE_LINK_PATH TRUE
				INSTALL_RPATH ${BLUEBILD_PYTHON_RPATH})
	endif()

	# destiantion dir must match module name in setup.py
	install(TARGETS pybluebild DESTINATION bluebild)
	install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/bluebild/__init__.py" DESTINATION bluebild)
	# if BLUEBILD_INSTALL is equal PYTHON, the library has not been added to the install target
	install(TARGETS bluebild DESTINATION bluebild/_libs)
endif()
