# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers
cmake_minimum_required(VERSION 3.1)
project(libimread-config)

set(libimread-config_VERSION_MAJOR "0")
set(libimread-config_VERSION_MINOR "2")
set(libimread-config_VERSION_PATCH "0")

find_path(LIBDOCOPT_INCLUDE_DIR NAMES docopt.h
                                PATHS ${DOCOPT_DIR})

include_directories(
    ${libimread_include_dir}
    ${LIBDOCOPT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

add_definitions(
    -Wall -Werror -O3
    -std=c++17 -stdlib=libc++
    -fstrict-aliasing
    -fstack-protector-all
    -mtune=native)

set(EXTRA_LIBS
    ${EXTRA_LIBS}
    imread_static
    docopt_static)

set(hdrs
    "detail.hh"
    "libimread-config.hh")
set(srcs
    "detail.cpp"
    "main.cpp")

add_executable("imread-config" ${srcs} ${hdrs})
target_link_libraries("imread-config" ${EXTRA_LIBS})
add_dependencies("imread-config" imread_static)
add_dependencies("imread-config" docopt_static)
set_property(
    TARGET "imread-config"
    PROPERTY POSITION_INDEPENDENT_CODE 1)


install(TARGETS "imread-config" DESTINATION bin)
