# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers

# The minimum CMake version required to build this project
cmake_minimum_required(VERSION 3.3)

# Set a new CMake project
project(libimread)

option(IM_USE_GCC           "Compile using GCC"                             OFF)
option(IM_RESYMBOLIZE       "Regenerate include/libimread/symbols.hpp"      OFF)
option(IM_GENERATE_HEADER   "Only generate libimread.hpp (do not compile)"  OFF)

if(IM_USE_GCC)
    # hardcode homebrew path for now
    set(CMAKE_C_COMPILER    "/usr/local/opt/gcc/bin/gcc-5")
    set(CMAKE_CXX_COMPILER  "/usr/local/opt/gcc/bin/g++-5")
endif()

# C++ standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
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
set(CMAKE_SYSTEM_LIBRARY_PATH ${CMAKE_SYSTEM_LIBRARY_PATH}
    $ENV{LD_LIBRARY_PATH}
    /usr/lib/system)

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
SET(IOD_SYMBOLIZE_DIR1  "${CMAKE_CURRENT_SOURCE_DIR}/python")
SET(IOD_SYMBOLIZE_DIR2  "${CMAKE_CURRENT_SOURCE_DIR}/src")
IOD_SYMBOLIZE(
    ${IOD_SYMBOLS_HEADER}
    ${IOD_SYMBOLIZE_DIR0}
    ${IOD_SYMBOLIZE_DIR1}
    ${IOD_SYMBOLIZE_DIR2})

SET(DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps")
SET(APPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/apps")
SET(IOD_DIR "${DEPS_DIR}/iod")
SET(IOD_TEST_DIR "${IOD_DIR}/tests")
SET(CROSSGUID_DIR "${DEPS_DIR}/crossguid")
SET(DOCOPT_DIR "${DEPS_DIR}/docopt")
SET(IMAGECOMPRESSION_DIR "${DEPS_DIR}/imagecompression")
SET(AFNETWORKING_DIR "${DEPS_DIR}/AFNetworking")
SET(MABLOCKCLOSURE_DIR "${DEPS_DIR}/MABlockClosure")
SET(SG14_DIR "${DEPS_DIR}/SG14")
SET(SSZIP_DIR "${DEPS_DIR}/SSZipArchive")
SET(LIBIMREAD_CONFIG_DIR "${APPS_DIR}/libimread-config")
SET(IMPASTE_DIR "${APPS_DIR}/impaste")
add_subdirectory(${DEPS_DIR})
set_property(DIRECTORY ${IOD_TEST_DIR}
             PROPERTY EXCLUDE_FROM_ALL TRUE)
set_property(DIRECTORY ${IOD_TEST_DIR}
             PROPERTY TEST)

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
find_package(LibUnwind REQUIRED)

set(HDF5_USE_STATIC_LIBRARIES false)
set(HDF5_ROOT /usr/local/opt/hdf5/)
find_package(HDF5 REQUIRED
    COMPONENTS C CXX HL)

# Add the project include directory
include_directories(${${PROJECT_NAME}_include_dir})
include_directories("${CMAKE_BINARY_DIR}/../include")
include_directories("${CMAKE_BINARY_DIR}")
include_directories("${CROSSGUID_DIR}")
include_directories("${DOCOPT_DIR}")
include_directories("${AFNETWORKING_DIR}")
include_directories("${MABLOCKCLOSURE_DIR}")
include_directories("${IOD_DIR}")
include_directories("${SG14_DIR}")
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
    ${HALIDE_INCLUDE_DIR}
    ${LIBUNWIND_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIRS})

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
# add_dependencies(libimread "project_header")
add_dependencies(imread "project_header")
# add_dependencies(libimread "libguid")
add_dependencies(imread "guid")
# add_dependencies(libimread "libimagecompression")
add_dependencies(imread "imagecompression")
# add_dependencies(libimread "libAFNetworking")
add_dependencies(imread "AFNetworking")
# add_dependencies(libimread "libMABlockClosure")
add_dependencies(imread "MABlockClosure")
# add_dependencies(libimread "libsszip")
# add_dependencies(imread "sszip")
# add_dependencies(libimread "iod_symbolize")
add_dependencies(imread "iod_symbolize")

# deal with config variables
get_property(IM_COMPILE_OPTIONS GLOBAL PROPERTY COMPILE_FLAGS)
get_target_property(IM_INCLUDE_DIRECTORIES imread INCLUDE_DIRECTORIES)
get_target_property(IM_LINK_LIBRARIES imread LINK_LIBRARIES)
get_target_property(IM_LINK_FLAGS imread LINK_FLAGS)

# Configure the project-settings header file
set(PROJECT_HEADER
    "${PROJECT_BINARY_DIR}/libimread/libimread.hpp")
configure_file(
    "${hdrs_dir}/libimread.hpp.in"
    ${PROJECT_HEADER})
add_custom_target("project_header"
    DEPENDS ${PROJECT_HEADER})
set_source_files_properties(
    ${PROJECT_HEADER}
    PROPERTIES GENERATED TRUE)

# target_link_libraries(libimread
#     libguid libdocopt
#     libMABlockClosure
#     libsszip
#     png z ${EXTRA_LIBS}
#     ${TIFF_LIBRARIES}
#     ${JPEG_LIBRARIES}
#     ${WEBP_LIBRARIES}
#     ${HALIDE_LIBRARIES}
#     ${HDF5_LIBRARIES})

target_link_libraries(imread
    png z guid
    MABlockClosure
    imagecompression
    ${EXTRA_LIBS}
    ${TIFF_LIBRARIES}
    ${JPEG_LIBRARIES}
    ${WEBP_LIBRARIES}
    ${HALIDE_LIBRARIES}
    ${HDF5_LIBRARIES})

# Add the apps
add_subdirectory(${APPS_DIR})

# Build the tests
add_subdirectory(${TEST_SOURCE_DIR})
add_custom_command(
    OUTPUT ${TEST_INCLUDE_DIR}/test_data.hpp
    COMMAND ${TEST_SCRIPT_DIR}/scan-test-data.py > ${TEST_INCLUDE_DIR}/test_data.hpp)
add_custom_target("test_data_header"
    DEPENDS ${TEST_INCLUDE_DIR}/test_data.hpp)
set_source_files_properties(
    ${TEST_INCLUDE_DIR}/test_data.hpp
    PROPERTIES GENERATED TRUE)
set(imread_tests "test_${PROJECT_NAME}")
add_executable(imread_tests ${TEST_SOURCES})
set_target_properties(imread_tests
    PROPERTIES LINK_FLAGS ${COMMON_LINK_FLAGS})
# add_dependencies("test_data_header" libimread)
add_dependencies("test_data_header" imread)
add_dependencies(imread_tests "project_header")
add_dependencies(imread_tests "test_data_header")
add_dependencies(imread_tests "iod_symbolize")
add_dependencies(imread_tests "guid")
add_dependencies(imread_tests "imagecompression")
add_dependencies(imread_tests "MABlockClosure")
add_dependencies(imread_tests "AFNetworking")
add_dependencies(imread_tests "sszip")
add_dependencies(imread_tests "imread")
link_directories(${CMAKE_BINARY_DIR})
target_link_libraries(imread_tests
    imread sszip
    AFNetworking
    MABlockClosure
    imagecompression
    guid png z
    ${EXTRA_LIBS}
    ${TIFF_LIBRARIES}
    ${JPEG_LIBRARIES}
    ${WEBP_LIBRARIES}
    ${HALIDE_LIBRARIES}
    ${HDF5_LIBRARIES})

# set up ctest and cdash
# … the add_imread_test() macro is defined in tests/CMakeLists.txt
enable_testing()
include(CTest)
add_imread_test("appkit-copy-paste")
add_imread_test("apple-io")
add_imread_test("blockhash")
add_imread_test("filesystem")
add_imread_test("gif-write")
add_imread_test("halide-io")
add_imread_test("hdf5-io")
add_imread_test("json-block-traverse")
add_imread_test("libguid")
add_imread_test("nsdictionary-options-map")
add_imread_test("nsurl-image-types")
add_imread_test("objc-rt")
# add_imread_test("refcount")
add_imread_test("sfinae")
add_imread_test("libsszip")
add_imread_test("terminator")
add_imread_test("interleaved-io")

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
