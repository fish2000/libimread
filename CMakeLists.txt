# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers

# The minimum CMake version required to build this project
cmake_minimum_required(VERSION 3.0)
# SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#                       ${CMAKE_SOURCE_DIR}/cmake)

SET(BREWED_CLANG_PATH /usr/local/opt/llvm)
# SET(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}
#                       ${BREWED_CLANG_PATH}
#                       /usr/local/opt/eigen)


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
SET(TEST_SCRIPT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/scripts")
SET(TEST_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/include")

SET(DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps")
SET(IOD_DIR "${DEPS_DIR}/iod")
SET(IOD_TEST_DIR "${IOD_DIR}/tests")
add_subdirectory(${DEPS_DIR})
set_property(DIRECTORY ${IOD_TEST_DIR}
             PROPERTY EXCLUDE_FROM_ALL TRUE)
set_property(DIRECTORY ${IOD_TEST_DIR}
             PROPERTY TEST)

execute_process(
    WORKING_DIRECTORY ${TEST_INCLUDE_DIR}
    COMMAND "rm -f ./test_data.hpp"
    COMMENT "Deleting generated test data header")
execute_process(
    WORKING_DIRECTORY "/tmp"
    COMMAND "rm -f /tmp/*.gif"
    COMMENT "Deleting generated temporary GIF files")

# Load the project configuration file. CMake will search in the directory setted
# above for a module file named libimreadConfig.cmake. The configuration
# file will set the different directories and libraries required by the library
find_package(${PROJECT_NAME} REQUIRED)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
find_package(TIFF REQUIRED)
find_package(WEBP REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Halide REQUIRED)
find_package(Eigen3 REQUIRED)

# Add the project include directory
include_directories(${${PROJECT_NAME}_include_dir})
include_directories("${BREWED_CLANG_PATH}/include")
include_directories("${CMAKE_BINARY_DIR}/../include")
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
    ${HALIDE_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR})

# Set the files and options required to build the library
include(fileList.cmake)

# Build the library
add_library(libimread STATIC ${srcs} ${hdrs})
add_library(imread SHARED ${srcs} ${hdrs})
set_target_properties(libimread
    PROPERTIES ARCHIVE_OUTPUT_NAME "imread")
set_target_properties(imread
    PROPERTIES LIBRARY_OUTPUT_NAME "imread")
set_target_properties(libimread
    PROPERTIES LINK_FLAGS ${COMMON_LINK_FLAGS})
set_target_properties(imread
    PROPERTIES LINK_FLAGS ${COMMON_LINK_FLAGS})
add_dependencies(libimread "libsszip")
add_dependencies(imread "libsszip")
add_dependencies(libimread "iod_generate_symbols")
add_dependencies(imread "iod_generate_symbols")
target_link_libraries(libimread
    libsszip
    png z ${EXTRA_LIBS}
    ${EIGEN3_LIBRARIES}
    ${TIFF_LIBRARIES}
    ${JPEG_LIBRARIES}
    ${WEBP_LIBRARIES}
    ${HALIDE_LIBRARIES})
target_link_libraries(imread
    sszip
    png z ${EXTRA_LIBS}
    ${EIGEN3_LIBRARIES}
    ${TIFF_LIBRARIES}
    ${JPEG_LIBRARIES}
    ${WEBP_LIBRARIES}
    ${HALIDE_LIBRARIES})

# Run the tests
add_subdirectory(${TEST_SOURCE_DIR})
add_custom_command(
    OUTPUT ${TEST_INCLUDE_DIR}/test_data.hpp
    COMMAND ${TEST_SCRIPT_DIR}/scan-test-data.py > ${TEST_INCLUDE_DIR}/test_data.hpp)
add_custom_target("test_data_header"
    DEPENDS ${TEST_INCLUDE_DIR}/test_data.hpp)
set_source_files_properties(
    ${TEST_INCLUDE_DIR}/test_data.hpp
    PROPERTIES GENERATED TRUE)
add_executable("test_${PROJECT_NAME}" ${TEST_SOURCES})
add_dependencies("test_${PROJECT_NAME}" "test_data_header")
add_dependencies("test_${PROJECT_NAME}" "libsszip")
add_dependencies("test_${PROJECT_NAME}" "iod_generate_symbols")
link_directories(${CMAKE_BINARY_DIR})
target_link_libraries("test_${PROJECT_NAME}"
    png z imread sszip
    ${EXTRA_LIBS}
    ${EIGEN3_LIBRARIES}
    ${TIFF_LIBRARIES}
    ${JPEG_LIBRARIES}
    ${WEBP_LIBRARIES}
    ${HALIDE_LIBRARIES})

install(TARGETS libimread DESTINATION lib)
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
