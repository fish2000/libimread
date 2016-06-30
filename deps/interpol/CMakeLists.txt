# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers
cmake_minimum_required(VERSION 3.0)
project(interpol LANGUAGES C)

set(interpol_VERSION_MAJOR "0")
set(interpol_VERSION_MINOR "9")
set(interpol_VERSION_PATCH "9")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR})

add_definitions(
    -Werror -Wall
    -x c -std=c11
    -funroll-loops
    -O3 -m64
    -mtune=native)

set(EXTRA_LIBS ${EXTRA_LIBS} m c)

set(srcs
    "io.c"
    "interpol.c"
    "coeff.c")
# "demo.c"

set(hdrs
    "io.h"
    "interpol.h"
    "coeff.h")

add_library(libinterpol STATIC ${srcs} ${hdrs})
add_library(interpol SHARED ${srcs} ${hdrs})

set_target_properties(libinterpol
    PROPERTIES OUTPUT_NAME "interpol")
set_target_properties(interpol
    PROPERTIES OUTPUT_NAME "interpol")
target_link_libraries(libinterpol ${EXTRA_LIBS})
target_link_libraries(interpol ${EXTRA_LIBS})

install(TARGETS libinterpol DESTINATION lib)
install(TARGETS interpol DESTINATION lib)
install(DIRECTORY ./ DESTINATION include/interpol
    FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ./ DESTINATION include/interpol
    FILES_MATCHING PATTERN "*.img")
