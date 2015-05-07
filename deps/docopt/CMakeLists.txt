# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers
cmake_minimum_required(VERSION 3.0)
project(libdocopt LANGUAGES C CXX)

# TODO: compile examples and tests (why not??)
# "main.cpp"
# "run_testcase.cpp"

set(libdocopt_VERSION_MAJOR "0")
set(libdocopt_VERSION_MINOR "6")
set(libdocopt_VERSION_PATCH "1")

add_definitions(
    -Wall -Werror -fstack-protector-all
    -std=c++14 -stdlib=libc++ -O3
    -mtune=native -fstrict-aliasing)

set(srcs
    "docopt.cpp")

set(hdrs
    "docopt.h"
    "docopt_private.h"
    "docopt_util.h"
    "docopt_value.h")

add_library(libdocopt STATIC ${srcs} ${hdrs})
add_library(docopt SHARED ${srcs} ${hdrs})

set_target_properties(libdocopt
    PROPERTIES ARCHIVE_OUTPUT_NAME "docopt")
set_target_properties(docopt
    PROPERTIES LIBRARY_OUTPUT_NAME "docopt")
target_link_libraries(libdocopt)
target_link_libraries(docopt)

install(TARGETS libdocopt DESTINATION lib)
install(TARGETS docopt DESTINATION lib)
install(DIRECTORY ./ DESTINATION include/libdocopt
    FILES_MATCHING PATTERN "*.h")
