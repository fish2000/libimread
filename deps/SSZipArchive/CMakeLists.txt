# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers
cmake_minimum_required(VERSION 3.0)
project(libsszip LANGUAGES C)

set(libsszip_VERSION_MAJOR "0")
set(libsszip_VERSION_MINOR "3")
set(libsszip_VERSION_PATCH "3")

find_package(ZLIB REQUIRED)
find_package(BZIP2)
find_path(FOUNDATION_INCLUDE_DIR Foundation/Foundation.h)
find_library(FOUNDATION_LIBRARY Foundation)

include_directories(
    ${ZLIB_INCLUDE_DIR}
    ${BZIP2_INCLUDE_DIR}
    ${FOUNDATION_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

add_definitions(
    -Wall -Werror -fstack-protector-all
    -fobjc-abi-version=3
    -fobjc-arc -std=c99
    -x objective-c
    -DHAVE_BZIP2 -DHAS_STDINT_H)

set(EXTRA_LIBS ${EXTRA_LIBS}
    ${FOUNDATION_LIBRARY}
    ${ZLIB_LIBRARY}
    ${BZIP2_LIBRARY})

set(srcs
    "SSZipArchive.m"
    "minizip/ioapi.c"
    "minizip/mztools.c"
    "minizip/unzip.c"
    "minizip/zip.c")

set(hdrs
    "SSZipArchive.h"
    "minizip/crypt.h"
    "minizip/ioapi.h"
    "minizip/mztools.h"
    "minizip/unzip.h"
    "minizip/zip.h")

# add_library(libsszip STATIC ${srcs} ${hdrs})
add_library(sszip SHARED ${srcs} ${hdrs})

# set_target_properties(libsszip
#     PROPERTIES ARCHIVE_OUTPUT_NAME "sszip")
set_target_properties(sszip
    PROPERTIES LIBRARY_OUTPUT_NAME "sszip")
# target_link_libraries(libsszip ${EXTRA_LIBS})
target_link_libraries(sszip ${EXTRA_LIBS})

# install(TARGETS libsszip DESTINATION lib)
install(TARGETS sszip DESTINATION lib)
install(DIRECTORY SSZipArchive/ DESTINATION include/libsszip
    FILES_MATCHING PATTERN "*.h")
install(DIRECTORY SSZipArchive/minizip/ DESTINATION include/libsszip/minizip
    FILES_MATCHING PATTERN "*.h")
