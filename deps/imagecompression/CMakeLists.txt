# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers
cmake_minimum_required(VERSION 3.0)
project(imagecompression LANGUAGES C CXX)

set(imagecompression_VERSION_MAJOR "0")
set(imagecompression_VERSION_MINOR "3")
set(imagecompression_VERSION_PATCH "5")

add_definitions(
    -DIS_LITTLE_ENDIAN
    -Wall -Werror
    -std=c++14 -stdlib=libc++ -O3
    -mtune=native -fstrict-aliasing)

set(COMMON_LINK_FLAGS
    -m64)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR})

set(hdrs
    "base/integral_types.h"
    "base/logging.h"
    "image_compression/internal/bit_util.h"
    "image_compression/internal/color_types.h"
    "image_compression/internal/compressor4x4_helper.h"
    "image_compression/internal/dxtc_const_color_table.h"
    "image_compression/internal/pixel4x4.h"
    "image_compression/public/compressed_image.h"
    "image_compression/public/compressor.h"
    "image_compression/public/dxtc_compressor.h"
    "image_compression/public/etc_compressor.h"
    "image_compression/public/pvrtc_compressor.h")

set(srcs
    "image_compression/internal/compressor4x4_helper.cc"
    "image_compression/internal/dxtc_compressor.cc"
    "image_compression/internal/dxtc_const_color_table.cc"
    "image_compression/internal/etc_compressor.cc"
    "image_compression/internal/pixel4x4.cc"
    "image_compression/internal/pvrtc_compressor.cc")

foreach(FLAG IN LISTS ${COMMON_LINK_FLAGS})

    set_source_files_properties(
        ${srcs}
        PROPERTIES LINK_FLAGS ${LINK_FLAGS} ${FLAG})

endforeach()

# add_library(libimagecompression STATIC ${srcs} ${hdrs})
add_library(imagecompression SHARED ${srcs} ${hdrs})

# set_target_properties(libimagecompression
#     PROPERTIES ARCHIVE_OUTPUT_NAME "guid")
set_target_properties(imagecompression
    PROPERTIES LIBRARY_OUTPUT_NAME "imagecompression")
# target_link_libraries(libimagecompression)
target_link_libraries(imagecompression)

# install(TARGETS libimagecompression DESTINATION lib)
install(TARGETS imagecompression DESTINATION lib)
install(DIRECTORY ./base/ DESTINATION include/imagecompression/base
    FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ./image_compression/internal/ DESTINATION include/imagecompression/internal
    FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ./image_compression/public/ DESTINATION include/imagecompression/public
    FILES_MATCHING PATTERN "*.h")
