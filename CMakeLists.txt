# Author: Félix C. Morency
# 2011.10

# The minimum CMake version required to build this project
cmake_minimum_required(VERSION 2.6)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      ${CMAKE_SOURCE_DIR}/cmake)

SET(BREWED_CLANG_PATH /usr/local/opt/llvm)
SET(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}
                      ${BREWED_CLANG_PATH}
                      /usr/local/opt/eigen)


SET(CMAKE_C_COMPILER    ${BREWED_CLANG_PATH}/bin/clang)
SET(CMAKE_CXX_COMPILER  ${BREWED_CLANG_PATH}/bin/clang++)

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
SET(TEST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests")

# Load the project configuration file. CMake will search in the directory setted
# above for a module file named libimreadConfig.cmake. The configuration
# file will set the different directories and libraries required by the library
find_package(${PROJECT_NAME} REQUIRED)

find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
find_package(TIFF REQUIRED)
find_package(WEBP REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Eigen3 REQUIRED)

# Add the project include directory
include_directories(${${PROJECT_NAME}_include_dir})
include_directories("${CMAKE_BINARY_DIR}/../include")
include_directories("${BREWED_CLANG_PATH}/include")
include_directories("${CMAKE_BINARY_DIR}")

# Ugh
link_directories("${BREWED_CLANG_PATH}/lib")
link_directories(/usr/local/lib)

include_directories(
    ${PNG_INCLUDE_DIR}
    ${JPEG_INCLUDE_DIR}
    ${TIFF_INCLUDE_DIR}
    ${ZLIB_INCLUDE_DIR}
    ${WEBP_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR})

#find_package(Iod REQUIRED)
#find_package(Vpp REQUIRED)
#include_directories(${IOD_INCLUDE_DIR})
#include_directories(${VPP_INCLUDE_DIR})
include_directories(/usr/local/opt/iod/include)
include_directories(/usr/local/opt/vpp/include)

find_package(Halide REQUIRED)
include_directories(${HALIDE_INCLUDE_DIR})

# Generate IOD symbol list
# file(GLOB formats ${CMAKE_CURRENT_SOURCE_DIR}/include/libimread/IO *.hh)
# add_custom_command(
#     OUTPUT ${PROJECT_BINARY_DIR}/libimread/symbols.hpp
#     COMMAND iod_generate_symbols ${formats} ${PROJECT_BINARY_DIR}/libimread/symbols.hpp)
# add_custom_target("iod_symbols"
#     DEPENDS ${HALIDE_INCLUDE_DIR}/libimread/symbols.hpp)
# set_source_files_properties(
#     ${PROJECT_BINARY_DIR}/libimread/symbols.hpp
#     PROPERTIES GENERATED TRUE)

# Set the source files required to build the library
include(fileList.cmake)

IF(APPLE)
    INCLUDE_DIRECTORIES(/Developer/Headers/FlatCarbon)
    FIND_LIBRARY(COCOA_LIBRARY Cocoa)
    FIND_LIBRARY(FOUNDATION_LIBRARY Foundation)
    FIND_LIBRARY(COREFOUNDATION_LIBRARY CoreFoundation)
    #FIND_LIBRARY(CARBON_LIBRARY Carbon)
    #FIND_LIBRARY(QUICKTIME_LIBRARY QuickTime)
    #FIND_LIBRARY(APP_SERVICES_LIBRARY ApplicationServices)
    MARK_AS_ADVANCED(COCOA_LIBRARY
                     FOUNDATION_LIBRARY
                     COREFOUNDATION_LIBRARY)
   SET(EXTRA_LIBS
       ${EXTRA_LIBS}
       ${COCOA_LIBRARY} ${FOUNDATION_LIBRARY}
       ${COREFOUNDATION_LIBRARY})
   # set_source_files_properties(GLOB_RECURSE "${srcs_dir}/*.m"
   #     PROPERTIES COMPILE_FLAGS ${COMPILE_FLAGS} -ObjC++)
   set_source_files_properties(GLOB_RECURSE "${srcs_dir}/*.mm"
       PROPERTIES COMPILE_FLAGS ${COMPILE_FLAGS} -ObjC++)
   set_source_files_properties(GLOB_RECURSE "${srcs_dir}/*.mm"
       PROPERTIES COMPILE_FLAGS ${COMPILE_FLAGS} -fobjc-arc)
ENDIF(APPLE)

add_definitions(-std=c++14 -stdlib=libc++ -O3)
# add_definitions(-isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.10.sdk)

# Build the library
#add_library(imread-static STATIC ${srcs} ${hdrs})
add_library(imread SHARED ${srcs} ${hdrs})
# add_dependencies(imread ${PROJECT_BINARY_DIR}/libimread/symbols.hpp)
target_link_libraries(imread
    png z ${EXTRA_LIBS}
    ${EIGEN3_LIBRARIES}
    ${TIFF_LIBRARIES}
    ${JPEG_LIBRARIES}
    ${WEBP_LIBRARIES}
    ${HALIDE_LIBRARIES})

# Run the tests
add_subdirectory(${TEST_SOURCE_DIR})
add_executable("test_${PROJECT_NAME}" ${TEST_SOURCES})
# add_dependencies("test_${PROJECT_NAME}" ${PROJECT_BINARY_DIR}/libimread/symbols.hpp)
link_directories(${CMAKE_BINARY_DIR})
target_link_libraries("test_${PROJECT_NAME}"
    png z imread ${EXTRA_LIBS}
    ${EIGEN3_LIBRARIES}
    ${TIFF_LIBRARIES}
    ${JPEG_LIBRARIES}
    ${WEBP_LIBRARIES}
    ${HALIDE_LIBRARIES})

#install(TARGETS imread-static DESTINATION lib)
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
