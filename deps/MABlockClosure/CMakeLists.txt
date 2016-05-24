# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers
cmake_minimum_required(VERSION 3.1)
project(MABlockClosure)

set(MABlockClosure_VERSION_MAJOR "0")
set(MABlockClosure_VERSION_MINOR "1")
set(MABlockClosure_VERSION_PATCH "0")

list(APPEND CMAKE_PREFIX_PATH /usr/local/opt/libffi/lib/pkgconfig)
include(FindLibFFI.cmake)
find_path(FOUNDATION_INCLUDE_DIR Foundation/Foundation.h)
find_library(FOUNDATION_LIBRARY Foundation)

include_directories(
    ${LIBFFI_INCLUDE_DIR}
    ${FOUNDATION_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

add_definitions(
    -fstack-protector-all
    -fobjc-abi-version=3
    -fno-objc-arc
    -std=c99
    -x objective-c)

set(EXTRA_LIBS ${EXTRA_LIBS}
    ${FOUNDATION_LIBRARY}
    ${LIBFFI_LIBRARY})

set(srcs
    "MABlockClosure.m"
    "BlockFptr.m")

set(hdrs
    "MABlockClosure.h"
    "BlockFptr.h"
    "BlockClosure.h")

add_library(MABlockClosure SHARED ${srcs} ${hdrs})

set_target_properties(MABlockClosure
    PROPERTIES OUTPUT_NAME "MABlockClosure")
target_link_libraries(MABlockClosure ${EXTRA_LIBS})

install(TARGETS MABlockClosure DESTINATION lib)
install(DIRECTORY ./ DESTINATION include/MABlockClosure
    FILES_MATCHING PATTERN "*.h")
