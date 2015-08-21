# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers
cmake_minimum_required(VERSION 3.1)
project(maBlockClosure)

set(maBlockClosure_VERSION_MAJOR "0")
set(maBlockClosure_VERSION_MINOR "1")
set(maBlockClosure_VERSION_PATCH "0")

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

# add_library(libmaBlockClosure STATIC ${srcs} ${hdrs})
add_library(maBlockClosure SHARED ${srcs} ${hdrs})

# set_target_properties(libmaBlockClosure
#     PROPERTIES OUTPUT_NAME "maBlockClosure")
set_target_properties(maBlockClosure
    PROPERTIES OUTPUT_NAME "maBlockClosure")
# target_link_libraries(libmaBlockClosure ${EXTRA_LIBS})
target_link_libraries(maBlockClosure ${EXTRA_LIBS})

# install(TARGETS libmaBlockClosure DESTINATION lib)
install(TARGETS maBlockClosure DESTINATION lib)
install(DIRECTORY MABlockClosure/ DESTINATION include/libmaBlockClosure
    FILES_MATCHING PATTERN "*.h")
