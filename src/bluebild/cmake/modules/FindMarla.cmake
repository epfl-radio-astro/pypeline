#.rst:
# FindMarla
# -----------
#
# This module looks for Marla library (https://gitlab.com/ursache/marla)
#
# The following variables are set
#
# ::
#
#   Marla_FOUND          - True if library is found
#   Marla_INCLUDE_DIRS   - The required include directory
#
# The following import target is created
#
# ::
#
#   Marla::Marla

# set paths to look for library
set(_Marla_PATHS ${MARLA_ROOT} $ENV{MARLA_ROOT})

find_path(Marla_INCLUDE_DIRS
    NAMES "floor.h"
    HINTS ${_Marla_PATHS}
    PATH_SUFFIXES "include"
)

# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Marla REQUIRED_VARS Marla_INCLUDE_DIRS)

# add target to link against
if(Marla_FOUND)
    if(NOT TARGET Marla::Marla)
        add_library(Marla::Marla INTERFACE IMPORTED)
    endif()
    set_property(TARGET Marla::Marla PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${Marla_INCLUDE_DIRS})
endif()

# prevent clutter in cache
MARK_AS_ADVANCED(Marla_FOUND Marla_INCLUDE_DIRS)
