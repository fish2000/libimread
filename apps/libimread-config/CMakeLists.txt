# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers
cmake_minimum_required(VERSION 3.1)
project(libimread-config)

set(libimread-config_VERSION_MAJOR "0")
set(libimread-config_VERSION_MINOR "1")
set(libimread-config_VERSION_PATCH "0")

find_path(LIBDOCOPT_INCLUDE_DIR NAMES docopt.h
                                PATHS ${DOCOPT_DIR})

include_directories(
    ${libimread_include_dir}
    ${LIBDOCOPT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

add_definitions(
    -Wall -Werror -O3
    -std=c++14 -stdlib=libc++
    -fstrict-aliasing
    -fstack-protector-all
    -mtune=native)

set(EXTRA_LIBS
    ${EXTRA_LIBS}
    imread_shared
    docopt_static)

set(hdrs "libimread-config.h")
set(srcs "main.cpp")

add_executable("imread-config" ${srcs} ${hdrs})
target_link_libraries("imread-config" ${EXTRA_LIBS})
add_dependencies("imread-config" imread_shared)
add_dependencies("imread-config" docopt_static)

install(TARGETS "imread-config" DESTINATION bin)
