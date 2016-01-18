# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers
cmake_minimum_required(VERSION 3.1)
project(impaste)

if(NOT APPLE)
    message(FATAL_ERROR "impaste is Apple-only, sorry, bailing out now!")
endif(NOT APPLE)

set(impaste_VERSION_MAJOR "0")
set(impaste_VERSION_MINOR "1")
set(impaste_VERSION_PATCH "0")

find_path(LIBDOCOPT_INCLUDE_DIR NAMES docopt.h
                                PATHS ${DOCOPT_DIR})

include_directories(
    ${libimread_include_dir}
    ${LIBDOCOPT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

add_definitions(
    -Wall -O3
    -std=c++14 -stdlib=libc++
    -fstrict-aliasing
    -fstack-protector-all
    -fobjc-arc
    -fobjc-abi-version=3
    -fobjc-call-cxx-cdtors
    -mtune=native
    -x objective-c++)

FIND_LIBRARY(COCOA_LIBRARY Cocoa)
FIND_LIBRARY(FOUNDATION_LIBRARY Foundation)
FIND_LIBRARY(COREFOUNDATION_LIBRARY CoreFoundation)
MARK_AS_ADVANCED(COCOA_LIBRARY
                FOUNDATION_LIBRARY
                COREFOUNDATION_LIBRARY)

set(EXTRA_LIBS
    ${EXTRA_LIBS}
    ${COCOA_LIBRARY}
    ${FOUNDATION_LIBRARY}
    ${COREFOUNDATION_LIBRARY}
    imread docopt)

set(hdrs "impaste.hh")
set(srcs "impaste.mm")
add_executable("impaste" ${srcs} ${hdrs})
target_link_libraries("impaste" ${EXTRA_LIBS})
add_dependencies("impaste" imread)
add_dependencies("impaste" docopt)
install(TARGETS "impaste" DESTINATION bin)
