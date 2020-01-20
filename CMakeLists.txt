# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

set (CMAKE_CXX_STANDARD 14)

find_package(PkgConfig REQUIRED)
pkg_search_module(OPENCV REQUIRED opencv)
pkg_search_module(OPENBLAS REQUIRED openblas)
pkg_search_module(NUMCPP REQUIRED NumCpp)
pkg_search_module(BOOST REQUIRED boost)

include_directories(include ${OPENCV_INCLUDE_DIRS} ${OPENBLAS_INCLUDE_DIRS} ${NUMCPP_INCLUDE_DIRS} ${BOOST_INCLUDE_DIRS})

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_executable(HeartBeat ${SOURCES} )

target_link_libraries( # Specifies the target library.
        HeartBeat

        # Links the target library to the log library
        # included in the NDK.
        ${log-lib}
        z
        ${OPENCV_LIBRARIES}
        ${BOOST_LIBRARIES}
        ${OPENBLAS_LIBRARIES}
        )
