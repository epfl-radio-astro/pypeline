# Install script for directory: /home/adas/bluebuild/pypeline/src/bluebild/python

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/adas/bluebuild/pypeline/src/bluebild/_skbuild/linux-x86_64-3.7/cmake-install/python")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "DEBUG")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bluebild/pybluebild.cpython-37m-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bluebild/pybluebild.cpython-37m-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bluebild/pybluebild.cpython-37m-x86_64-linux-gnu.so"
         RPATH "$ORIGIN/_libs")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bluebild" TYPE MODULE FILES "/home/adas/bluebuild/pypeline/src/bluebild/_skbuild/linux-x86_64-3.7/cmake-build/python/bluebild/pybluebild.cpython-37m-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bluebild/pybluebild.cpython-37m-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bluebild/pybluebild.cpython-37m-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bluebild/pybluebild.cpython-37m-x86_64-linux-gnu.so"
         OLD_RPATH "/home/adas/bluebuild/pypeline/src/bluebild/_skbuild/linux-x86_64-3.7/cmake-build/src:"
         NEW_RPATH "$ORIGIN/_libs")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bluebild/pybluebild.cpython-37m-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bluebild" TYPE FILE FILES "/home/adas/bluebuild/pypeline/src/bluebild/python/bluebild/__init__.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bluebild/_libs/libbluebild.so.0.1.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bluebild/_libs/libbluebild.so.1"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bluebild/_libs" TYPE SHARED_LIBRARY FILES
    "/home/adas/bluebuild/pypeline/src/bluebild/_skbuild/linux-x86_64-3.7/cmake-build/src/libbluebild.so.0.1.0"
    "/home/adas/bluebuild/pypeline/src/bluebild/_skbuild/linux-x86_64-3.7/cmake-build/src/libbluebild.so.1"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bluebild/_libs/libbluebild.so.0.1.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bluebild/_libs/libbluebild.so.1"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "/home/adas/bluebuild/finufft/lib:/home/adas/bluebuild/cufinufft/lib:"
           NEW_RPATH "")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bluebild/_libs/libbluebild.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bluebild/_libs/libbluebild.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bluebild/_libs/libbluebild.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bluebild/_libs" TYPE SHARED_LIBRARY FILES "/home/adas/bluebuild/pypeline/src/bluebild/_skbuild/linux-x86_64-3.7/cmake-build/src/libbluebild.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bluebild/_libs/libbluebild.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bluebild/_libs/libbluebild.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bluebild/_libs/libbluebild.so"
         OLD_RPATH "/home/adas/bluebuild/finufft/lib:/home/adas/bluebuild/cufinufft/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bluebild/_libs/libbluebild.so")
    endif()
  endif()
endif()

