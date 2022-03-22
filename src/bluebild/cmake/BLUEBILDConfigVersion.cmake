
# Prefer shared library
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/BLUEBILDSharedConfigVersion.cmake")
	include("${CMAKE_CURRENT_LIST_DIR}/BLUEBILDSharedConfigVersion.cmake")
else()
	include("${CMAKE_CURRENT_LIST_DIR}/BLUEBILDStaticConfigVersion.cmake")
endif()
