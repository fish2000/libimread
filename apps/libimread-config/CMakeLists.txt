# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers
cmake_minimum_required(VERSION 3.1)
project(libimread-config)

set(libimread-config_VERSION_MAJOR "0")
set(libimread-config_VERSION_MINOR "1")
set(libimread-config_VERSION_PATCH "0")

find_path(LIBDOCOPT_INCLUDE_DIR NAMES docopt.h
                                PATHS ${DOCOPT_DIR})
# find_library(LIBDOCOPT_LIBRARY docopt)

include_directories(
    ${libimread_include_dir}
    ${LIBDOCOPT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

add_definitions(
    -Wall -Werror -fstack-protector-all
    -std=c++14 -stdlib=libc++ -O3
    -mtune=native -fstrict-aliasing)

set(EXTRA_LIBS
    ${EXTRA_LIBS}
    imread docopt)

set(srcs "main.cpp")
add_executable("imread-config" ${srcs})
target_link_libraries("imread-config" ${EXTRA_LIBS})
add_dependencies("imread-config" imread)
install(TARGETS "imread-config" DESTINATION bin)
