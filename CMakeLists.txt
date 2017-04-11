# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers

# The minimum CMake version required to build this project
cmake_minimum_required(VERSION 3.3)

# Set a new CMake project
project(libimread)

option(IM_TESTS             "Compile and run tests"                             ON)
option(IM_APPS              "Compile and run companion apps"                    ON)
option(IM_USE_GCC           "Compile using GCC"                                 OFF)
option(IM_RESYMBOLIZE       "Regenerate include/libimread/symbols.hpp"          OFF)
option(IM_GENERATE_HEADER   "Only generate libimread.hpp (do not compile)"      OFF)
option(IM_COLOR_TRACE       "Use ANSI color in debug and error trace output"    ON)
option(IM_VERBOSE           "Print (highly nerd-oriented) verbose debug output" ON)
option(IM_TERMINATOR        "Use a libunwind-based termination handler"         ON)

unset(HAVE_AUTOFS_NOWAIT CACHE)
unset(HAVE_AUTOFS_NOTRIGGER CACHE)

if(IM_USE_GCC)
    # hardcode homebrew path for now
    set(CMAKE_C_COMPILER    "/usr/local/opt/gcc/bin/gcc-5")
    set(CMAKE_CXX_COMPILER  "/usr/local/opt/gcc/bin/g++-5")
endif()

# C++ standard
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)
# if(NOT CMAKE_CXX_STANDARD)
#     set(CMAKE_CXX_STANDARD 17)
# endif()

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

# Include the HalideGenerator.cmake library -- exposing the cmake function
# halide_add_generator_dependency() allowing Halide generator use in-project.
include(HalideProject)
include(HalideGenerator)

# Include IodSymbolize.cmake from the iod-symbolizer Python tool,
# exposing the cmake function IOD_SYMBOLIZE()
include(IodSymbolize)

# Set the location of the library configuration file if it has not already been
# set. This allow the library to be used by an external project without
# overwriting the variable.
if_defined_set(${PROJECT_NAME}_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
SET(TEST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests")
SET(TEST_SCRIPT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/scripts")
SET(TEST_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/include")

# Set up the directories to feed to the IOD symbolizer,
# and the target output header file to generate
SET(IOD_SYMBOLS_HEADER  "${CMAKE_CURRENT_SOURCE_DIR}/include/libimread/symbols.hpp")
SET(IOD_SYMBOLIZE_DIR0  "${CMAKE_CURRENT_SOURCE_DIR}/include/libimread")
SET(IOD_SYMBOLIZE_DIR1  "${CMAKE_CURRENT_SOURCE_DIR}/python")
SET(IOD_SYMBOLIZE_DIR2  "${CMAKE_CURRENT_SOURCE_DIR}/src")
IOD_SYMBOLIZE(
    ${IOD_SYMBOLS_HEADER}
    ${IOD_SYMBOLIZE_DIR0}
    ${IOD_SYMBOLIZE_DIR1}
    ${IOD_SYMBOLIZE_DIR2})

# Set variables for the deps and apps directories:
SET(APPS_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/apps")
SET(CAPN_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/capn")
SET(DEPS_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/deps")
SET(HALOGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/halogen")

# Set variables for the individual deps directories:
SET(IOD_DIR "${DEPS_DIR}/iod")
SET(IOD_TEST_DIR "${IOD_DIR}/tests")
SET(CROSSGUID_DIR "${DEPS_DIR}/crossguid")
SET(DOCOPT_DIR "${DEPS_DIR}/docopt")
# SET(INTERPOL_DIR "${DEPS_DIR}/interpol")
SET(IMAGECOMPRESSION_DIR "${DEPS_DIR}/imagecompression")
# SET(LIBBF_DIR "${DEPS_DIR}/libbf/src")
SET(LMDBXX_DIR "${DEPS_DIR}/lmdbxx")
SET(LRUCACHE11_DIR "${DEPS_DIR}/lrucache11")
SET(SG14_DIR "${DEPS_DIR}/SG14")
# SET(VDB_DIR "${DEPS_DIR}/vdb")

# Set variables for directories for the individual apps:
SET(LIBIMREAD_CONFIG_DIR "${APPS_DIR}/libimread-config")

# Add DEPS_DIR as a subdirectory ...
add_subdirectory(${DEPS_DIR})

# ... but exclude the IOD test directory from being built:
set_property(DIRECTORY ${IOD_TEST_DIR}
             PROPERTY EXCLUDE_FROM_ALL TRUE)
set_property(DIRECTORY ${IOD_TEST_DIR}
             PROPERTY TEST)

# Add HALOGEN_DIR -- set up Halide generators:
# add_subdirectory(${HALOGEN_DIR})

# Load the project configuration file. CMake will search in the directory setted
# above for a module file named libimreadConfig.cmake. The configuration
# file will set the different directories and libraries required by the library:
find_package(${PROJECT_NAME} REQUIRED)

# Find the required dependency libraries:
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
find_package(TIFF REQUIRED)
find_package(WEBP REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Halide REQUIRED)
find_package(LibUnwind REQUIRED)
find_package(CapnProto REQUIRED)

set(HDF5_USE_STATIC_LIBRARIES false)
# set(HDF5_ROOT /usr/local/opt/hdf5/)
# set(HDF5_DIR  /usr/local/opt/hdf5/)
set(HDF5_NO_FIND_PACKAGE_CONFIG_FILE true)
set(HDF5_COMPILER_NO_INTERROGATE true)

# message(STATUS "find_package(): looking for HDF5 within HDF5_ROOT = ${HDF5_ROOT}")
# find_package(HDF5 COMPONENTS C CXX HL NO_MODULE)
# if(NOT HDF5_FOUND)
#     message(STATUS "find_package(): couldn't find HDF5 cmake module configuration in ${HDF5_ROOT}")
#     message(STATUS "find_package(): falling back to cmake/FindHDF5.cmake")
#     include(FindHDF5)
#     find_package(HDF5 REQUIRED COMPONENTS C CXX HL)
# endif()

# include(FindHDF5)
find_package(HDF5 REQUIRED COMPONENTS C CXX HL)
# find_package(HDF5 REQUIRED COMPONENTS CXX)
# find_package(HDF5 REQUIRED COMPONENTS HL)

include(FindRocksDB)
# find_package(RocksDB REQUIRED)

# Include the project-specific `include` directories:
include_directories(${${PROJECT_NAME}_include_dir})
include_directories("${CMAKE_BINARY_DIR}/../include")
include_directories("${CMAKE_BINARY_DIR}")

# Include the `deps` library `include` directories:
include_directories("${CROSSGUID_DIR}")
include_directories("${DOCOPT_DIR}")
include_directories("${IMAGECOMPRESSION_DIR}")
# include_directories("${INTERPOL_DIR}")
# include_directories("${LIBBF_DIR}")
include_directories("${LMDBXX_DIR}")
include_directories("${LRUCACHE11_DIR}")
include_directories("${IOD_DIR}")
include_directories("${SG14_DIR}")
# include_directories("${VDB_DIR}/lib")

# Ugh
link_directories(/usr/lib)
link_directories(/usr/local/lib)

# Include the required dependency `include` directories:
include_directories(
    ${PNG_INCLUDE_DIR}
    ${JPEG_INCLUDE_DIR}
    ${TIFF_INCLUDE_DIR}
    ${ZLIB_INCLUDE_DIR}
    ${WEBP_INCLUDE_DIR}
    ${HALIDE_INCLUDE_DIR}
    ${LIBUNWIND_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIRS})

# Set the source files and source-file-specific options
# required to build the library:
include(CMakeProjectFiles.cmake)

# Actually build the library: set up an `OBJECT` target ...
add_library(imread OBJECT ${srcs} ${hdrs})
set_target_properties(imread
    PROPERTIES LIBRARY_OUTPUT_NAME "imread")
add_dependencies(imread "project_header")
add_dependencies(imread "guid")
add_dependencies(imread "imagecompression")
# add_dependencies(imread "interpol")
# add_dependencies(imread "libbf")
# add_dependencies(imread "vdb")
# add_dependencies(imread "vdb-client-shared")
add_dependencies(imread "iod_symbolize")

# ... and build shared and static target libraries,
# based on the `OBJECT` target:
set_property(
    TARGET imread
    PROPERTY POSITION_INDEPENDENT_CODE 1)
add_library(imread_shared SHARED $<TARGET_OBJECTS:imread>)
add_library(imread_static STATIC $<TARGET_OBJECTS:imread>)
set_target_properties(imread_shared
    PROPERTIES LIBRARY_OUTPUT_NAME "imread")
set_target_properties(imread_static
    PROPERTIES ARCHIVE_OUTPUT_NAME "imread")

# Actually build the library II: electric boogaloo
# ... these two target_link_libraries() calls set up the actual
# for-reals building up in all this:
target_link_libraries(imread_shared
    z png guid
    # libbf interpol
    imagecompression
    ${EXTRA_LIBS}
    ${TIFF_LIBRARIES}
    ${JPEG_LIBRARIES}
    ${WEBP_LIBRARIES}
    ${HALIDE_LIBRARIES}
    ${HDF5_C_LIBRARIES}
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_HL_LIBRARIES}
    ${CAPNP_LIBRARIES_LITE}
    ${ROCKSDB_LIBRARIES})

target_link_libraries(imread_static
    z png guid
    # libbf interpol
    imagecompression
    ${EXTRA_LIBS}
    ${TIFF_LIBRARIES}
    ${JPEG_LIBRARIES}
    ${WEBP_LIBRARIES}
    ${HALIDE_LIBRARIES}
    ${HDF5_C_LIBRARIES}
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_HL_LIBRARIES}
    ${CAPNP_LIBRARIES_LITE}
    ${ROCKSDB_LIBRARIES})

# Set and pass on compile-time config variable values:
get_property(IM_COMPILE_OPTIONS GLOBAL PROPERTY COMPILE_FLAGS)
get_target_property(IM_INCLUDE_DIRECTORIES imread INCLUDE_DIRECTORIES)
get_target_property(IM_LINK_LIBRARIES imread_shared LINK_LIBRARIES)
get_target_property(IM_LINK_FLAGS imread_shared LINK_FLAGS)

# set stuff from cache
set(IM_HAVE_STRINGVIEW_HH       ${HAVE_STRINGVIEW_HH})
set(IM_HAVE_SYS_SENDFILE_H      ${HAVE_SYS_SENDFILE_H})
set(IM_HAVE_AUTOFS_NOWAIT       ${HAVE_AUTOFS_NOWAIT})
set(IM_HAVE_AUTOFS_NOTRIGGER    ${HAVE_AUTOFS_NOTRIGGER})

# Configure and generate the compile-time project header file:
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

# Add the apps subdirectory, if we're building apps:
if(IM_APPS)
    add_subdirectory(${APPS_DIR})
endif(IM_APPS)

# Build the tests, if we're building tests:
if(IM_TESTS)

    # Bring in the tests subdirectory --
    # ... this defines add_imread_test(), see below:
    add_subdirectory(${TEST_SOURCE_DIR})
    
    # add_custom_command(
    #     OUTPUT ${APPS_DIR}/TestDataViewer/TestDataViewer/TestData.plist
    #     COMMAND ${TEST_SCRIPT_DIR}/generate-test-filemap.py > ${APPS_DIR}/TestDataViewer/TestDataViewer/TestData.plist)
    # add_custom_target("test_data_plist"
    #     DEPENDS ${APPS_DIR}/TestDataViewer/TestDataViewer/TestData.plist)
    # set_target_properties("test_data_plist"
    #     PROPERTIES GENERATED TRUE)
    # set_target_properties("test_data_plist"
    #     PROPERTIES MACOSX_BUNDLE_INFO_PLIST ${APPS_DIR}/TestDataViewer/TestDataViewer/TestData.plist)
    # add_dependencies("test_data_plist" imread)
    
    # Generate the test data header, if necessary:
    add_custom_command(
        OUTPUT ${TEST_INCLUDE_DIR}/test_data.hpp
        COMMAND ${TEST_SCRIPT_DIR}/generate-test-header.py > ${TEST_INCLUDE_DIR}/test_data.hpp)
    add_custom_target("test_data_header"
        DEPENDS ${TEST_INCLUDE_DIR}/test_data.hpp)
    set_source_files_properties(
        ${TEST_INCLUDE_DIR}/test_data.hpp
        PROPERTIES GENERATED TRUE)
    
    # Set up the `imread_tests` dependencies
    set(imread_tests "test_${PROJECT_NAME}")
    add_executable(imread_tests ${TEST_SOURCES})
    add_dependencies("test_data_header" imread)
    add_dependencies(imread_tests "project_header")
    add_dependencies(imread_tests "test_data_header")
    add_dependencies(imread_tests "iod_symbolize")
    add_dependencies(imread_tests "guid")
    add_dependencies(imread_tests "imagecompression")
    # add_dependencies(imread_tests "interpol")
    # add_dependencies(imread_tests "libbf")
    # add_dependencies(imread_tests "vdb")
    # add_dependencies(imread_tests "vdb-client-shared")
    add_dependencies(imread_tests "imread")
    link_directories(${CMAKE_BINARY_DIR})
    
    # Link the `imread_tests` executable
    target_link_libraries(imread_tests
        imread_shared
        imagecompression
        # libbf interpol
        # vdb-client
        guid png z
        ${EXTRA_LIBS}
        ${TIFF_LIBRARIES}
        ${JPEG_LIBRARIES}
        ${WEBP_LIBRARIES}
        ${HALIDE_LIBRARIES}
        ${HDF5_C_LIBRARIES}
        ${HDF5_CXX_LIBRARIES}
        ${HDF5_HL_LIBRARIES}
        ${CAPNP_LIBRARIES_LITE}
        ${ROCKSDB_LIBRARIES})
    
    # Set up ctest and cdash:
    enable_testing()
    include(CTest)
    
    # Set up individual test suites --
    # … the add_imread_test() macro is defined in tests/CMakeLists.txt:
    add_imread_test("blockhash")
    add_imread_test("byte-source-gzio")
    add_imread_test("byte-source-iterators")
    add_imread_test("environment")
    add_imread_test("filesystem")
    add_imread_test("attributes") # filesystem-attributes
    add_imread_test("execute") # filesystem-execute
    add_imread_test("gif-write")
    add_imread_test("halide-io")
    add_imread_test("hdf5-io")
    add_imread_test("imageformat-options")
    add_imread_test("imageview")
    add_imread_test("libguid")
    add_imread_test("refcount")
    add_imread_test("rocksdb")
    add_imread_test("sfinae")
    add_imread_test("terminator")
    add_imread_test("tif-write-multi")
    add_imread_test("interleaved-io")

endif(IM_TESTS)

# Install the built libraries and header files, as appropriate:
install(TARGETS imread_shared DESTINATION lib)
install(TARGETS imread_static DESTINATION lib)
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
