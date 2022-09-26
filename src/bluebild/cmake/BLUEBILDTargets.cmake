
# Prefer shared library
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/BLUEBILDSharedTargets.cmake")
	include("${CMAKE_CURRENT_LIST_DIR}/BLUEBILDSharedTargets.cmake")
else()
	include("${CMAKE_CURRENT_LIST_DIR}/BLUEBILDStaticTargets.cmake")
endif()
