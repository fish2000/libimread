# Author: Félix C. Morency
# 2011.10

# The minimum CMake version required to build this project
cmake_minimum_required(VERSION 2.6)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      ${CMAKE_SOURCE_DIR}/cmake)

# Set a new CMake project
project(libimread)

# Go through some stuff
set(libimread_VERSION_MAJOR "0")
set(libimread_VERSION_MINOR "1")
set(libimread_VERSION_PATCH "2")

# If the build script is called from a parent project,
# use the configuration from there.
if(NOT COMMAND if_defined_set)
    set(cmake_directory ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
endif()

# Set the CMake module path to the project cmake directory. This location will
# first be searched for external CMake packages.
set(CMAKE_MODULE_PATH ${cmake_directory})

# Include the library build configuration. This will also include the custom
# macro defined for the project.
include(build_config)

# Set the location of the library configuration file if it has not already been
# set. This allow the library to be used by an external project without
# overwritting the variable.
if_defined_set(${PROJECT_NAME}_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Load the project configuration file. CMake will search in the directory setted
# above for a module file named libimreadConfig.cmake. The configuration
# file will set the different directories and libraries required by the library
find_package(${PROJECT_NAME} REQUIRED)

find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
find_package(TIFF REQUIRED)
find_package(WEBP REQUIRED)
find_package(ZLIB REQUIRED)

# Add the project include directory
include_directories(${${PROJECT_NAME}_include_dir})
include_directories("${CMAKE_BINARY_DIR}/../include")
include_directories("${CMAKE_BINARY_DIR}")
link_directories(/usr/local/lib)

include_directories(
    ${PNG_INCLUDE_DIR}
    ${JPEG_INCLUDE_DIR}
    ${TIFF_INCLUDE_DIR}
    ${ZLIB_INCLUDE_DIR}
    ${WEBP_INCLUDE_DIR})

find_package(Halide REQUIRED)
find_package(JeMalloc REQUIRED)
include_directories(${HALIDE_INCLUDE_DIR})
include_directories(${JEMALLOC_INCLUDE_DIR})

# Set the source files required to build the library
include(fileList.cmake)

add_definitions(-std=c++1y -stdlib=libc++)

# Build the library
add_library(imread-static STATIC ${srcs} ${hdrs})
add_library(imread SHARED ${srcs} ${hdrs})
target_link_libraries(imread
    png z
    ${TIFF_LIBRARIES}
    ${JPEG_LIBRARIES}
    ${WEBP_LIBRARIES})

install(TARGETS imread-static DESTINATION lib)
install(TARGETS imread DESTINATION lib)
install(DIRECTORY ${PROJECT_BINARY_DIR}/libimread/ DESTINATION include/libimread
  FILES_MATCHING PATTERN "libimread.hpp")
install(DIRECTORY include/libimread/private/ DESTINATION include/libimread/private
  FILES_MATCHING PATTERN "*.h")
install(DIRECTORY include/libimread/ DESTINATION include/libimread
  FILES_MATCHING PATTERN "*.h")
install(DIRECTORY include/libimread/ DESTINATION include/libimread
  FILES_MATCHING PATTERN "*.hh")
install(DIRECTORY include/libimread/ DESTINATION include/libimread
  FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY cmake/ DESTINATION share/libimread
  FILES_MATCHING PATTERN "*libimreadConfig.cmake")
