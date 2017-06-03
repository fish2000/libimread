# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers
cmake_minimum_required(VERSION 3.0)
project(imagecompression LANGUAGES C CXX)

set(imagecompression_VERSION_MAJOR "0")
set(imagecompression_VERSION_MINOR "3")
set(imagecompression_VERSION_PATCH "5")

# Compilation options:
add_definitions(
    -DIS_LITTLE_ENDIAN
    -Wall -Werror
    -std=c++14 -stdlib=libc++ -O3
    -mtune=native -fstrict-aliasing)

# Linker options:
set(COMMON_LINK_FLAGS -m64)

# Include directories:
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Project header files:
set(hdrs "image_compression/internal/logging.h"
         "image_compression/internal/bit_util.h"
         "image_compression/internal/color_types.h"
         "image_compression/internal/compressor4x4_helper.h"
         "image_compression/internal/dxtc_const_color_table.h"
         "image_compression/internal/pixel4x4.h"
         "image_compression/public/compressed_image.h"
         "image_compression/public/compressor.h"
         "image_compression/public/dxtc_compressor.h"
         "image_compression/public/dxtc_to_etc_transcoder.h"
         "image_compression/public/etc_compressor.h"
         "image_compression/public/pvrtc_compressor.h")

# Project C++ source files:
set(srcs "image_compression/internal/compressor4x4_helper.cc"
         "image_compression/internal/dxtc_compressor.cc"
         "image_compression/internal/dxtc_const_color_table.cc"
         "image_compression/internal/dxtc_to_etc_transcoder.cc"
         "image_compression/internal/etc_compressor.cc"
         "image_compression/internal/pixel4x4.cc"
         "image_compression/internal/pvrtc_compressor.cc")

# Distribute linker options on a per-file basis:
foreach(FLAG IN LISTS ${COMMON_LINK_FLAGS})

    set_source_files_properties(
        ${srcs}
        PROPERTIES LINK_FLAGS ${LINK_FLAGS} ${FLAG})

endforeach()

# Add an “OBJECT” library target called “imagecompression”:
add_library(
    imagecompression
    OBJECT ${srcs} ${hdrs})
set_target_properties(
    imagecompression
    PROPERTIES LIBRARY_OUTPUT_NAME "imagecompression")
set_property(
    TARGET imagecompression
    PROPERTY POSITION_INDEPENDENT_CODE 1)

# Add shared and static library targets, based on the “OBJECT” library,
# respectively  called “imagecompression_shared” and “imagecompression_static”:
add_library(
    imagecompression_shared
    SHARED $<TARGET_OBJECTS:imagecompression>)
add_library(
    imagecompression_static
    STATIC $<TARGET_OBJECTS:imagecompression>)
set_target_properties(
    imagecompression_shared
    PROPERTIES LIBRARY_OUTPUT_NAME "imagecompression")
set_target_properties(
    imagecompression_static
    PROPERTIES ARCHIVE_OUTPUT_NAME "imagecompression")
target_link_libraries(imagecompression_shared)
target_link_libraries(imagecompression_static)

# Install the shared and static library target outputs:
install(TARGETS imagecompression_shared
        DESTINATION lib)
install(TARGETS imagecompression_static
        DESTINATION lib)

# Install the header files:
install(DIRECTORY   ./image_compression/internal/
        DESTINATION include/imagecompression/internal
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY   ./image_compression/public/
        DESTINATION include/imagecompression/public
        FILES_MATCHING PATTERN "*.h")
