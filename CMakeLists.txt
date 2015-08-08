# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers

# The minimum CMake version required to build this project
cmake_minimum_required(VERSION 3.0)

# Set a new CMake project
project(libimread)

option(IM_USE_GCC           "Compile using GCC"                             OFF)
option(IM_RESYMBOLIZE       "Regenerate include/libimread/symbols.hpp"      OFF)

if(IM_USE_GCC)
    # hardcode homebrew path for now
    set(CMAKE_C_COMPILER    "/usr/local/opt/gcc/bin/gcc-5")
    set(CMAKE_CXX_COMPILER  "/usr/local/opt/gcc/bin/g++-5")
endif()

if(IM_RESYMBOLIZE)
    execute_process(
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/libimread/"
        COMMAND "rm symbols.hpp"
        COMMENT "Deleting generated IOD symbol header file")
endif()

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

include(IodSymbolize)
SET(IOD_SYMBOLS_HEADER  "${CMAKE_CURRENT_SOURCE_DIR}/include/libimread/symbols.hpp")
SET(IOD_SYMBOLIZE_DIR0  "${CMAKE_CURRENT_SOURCE_DIR}/include/libimread")
SET(IOD_SYMBOLIZE_DIR1  "${CMAKE_CURRENT_SOURCE_DIR}/src")
IOD_SYMBOLIZE(
    ${IOD_SYMBOLIZE_DIR0}
    ${IOD_SYMBOLIZE_DIR1}
    ${IOD_SYMBOLS_HEADER})

SET(DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps")
SET(LEXICAL_CAST_DIR "${DEPS_DIR}/lexical_cast")
SET(IOD_DIR "${DEPS_DIR}/iod")
SET(IOD_TEST_DIR "${IOD_DIR}/tests")
SET(DOCOPT_DIR "${DEPS_DIR}/docopt")
SET(SSZIP_DIR "${DEPS_DIR}/SSZipArchive")
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

# Add the project include directory
include_directories(${${PROJECT_NAME}_include_dir})
include_directories("${LEXICAL_CAST_DIR}/include")
include_directories("${CMAKE_BINARY_DIR}/../include")
include_directories("${CMAKE_BINARY_DIR}")
include_directories("${DOCOPT_DIR}")
include_directories("${IOD_DIR}")
include_directories("${SSZIP_DIR}")

# Ugh
link_directories(/usr/lib)
link_directories(/usr/local/lib)

include_directories(
    ${PNG_INCLUDE_DIR}
    ${JPEG_INCLUDE_DIR}
    ${TIFF_INCLUDE_DIR}
    ${ZLIB_INCLUDE_DIR}
    ${WEBP_INCLUDE_DIR}
    ${HALIDE_INCLUDE_DIR})

# Set the files and options required to build the library
include(CMakeProjectFiles.cmake)

# Build the library
# add_library(libimread STATIC ${srcs} ${hdrs})
add_library(imread SHARED ${srcs} ${hdrs})
# set_target_properties(libimread
#     PROPERTIES ARCHIVE_OUTPUT_NAME "imread")
set_target_properties(imread
    PROPERTIES LIBRARY_OUTPUT_NAME "imread")
# set_target_properties(libimread
#     PROPERTIES LINK_FLAGS ${COMMON_LINK_FLAGS})
set_target_properties(imread
    PROPERTIES LINK_FLAGS ${COMMON_LINK_FLAGS})
# add_dependencies(libimread "libdocopt")
add_dependencies(imread "docopt")
# add_dependencies(libimread "libsszip")
add_dependencies(imread "sszip")
# add_dependencies(libimread  "iod_symbolize")
add_dependencies(imread     "iod_symbolize")

# target_link_libraries(libimread
#     libsszip
#     png z ${EXTRA_LIBS}
#     ${TIFF_LIBRARIES}
#     ${JPEG_LIBRARIES}
#     ${WEBP_LIBRARIES}
#     ${HALIDE_LIBRARIES})
target_link_libraries(imread
    sszip docopt
    png z ${EXTRA_LIBS}
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
add_dependencies("test_${PROJECT_NAME}" "iod_symbolize")
add_dependencies("test_${PROJECT_NAME}" "docopt")
add_dependencies("test_${PROJECT_NAME}" "sszip")
link_directories(${CMAKE_BINARY_DIR})
target_link_libraries("test_${PROJECT_NAME}"
    png z imread sszip docopt
    ${EXTRA_LIBS}
    ${TIFF_LIBRARIES}
    ${JPEG_LIBRARIES}
    ${WEBP_LIBRARIES}
    ${HALIDE_LIBRARIES})

# install(TARGETS libimread DESTINATION lib)
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
