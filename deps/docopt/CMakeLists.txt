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

# Add an “OBJECT” library target called “docopt”:
add_library(
    docopt
    OBJECT ${srcs} ${hdrs})
set_target_properties(
    docopt
    PROPERTIES LIBRARY_OUTPUT_NAME "docopt")
set_property(
    TARGET docopt
    PROPERTY POSITION_INDEPENDENT_CODE 1)

# Add shared and static library targets, based on the “OBJECT” library,
# respectively  called “docopt_shared” and “docopt_static”:
add_library(
    docopt_shared
    SHARED $<TARGET_OBJECTS:docopt>)
add_library(
    docopt_static
    STATIC $<TARGET_OBJECTS:docopt>)
set_target_properties(
    docopt_shared
    PROPERTIES LIBRARY_OUTPUT_NAME "docopt")
set_target_properties(
    docopt_static
    PROPERTIES ARCHIVE_OUTPUT_NAME "docopt")
target_link_libraries(docopt_shared)
target_link_libraries(docopt_static)

# Install the shared and static library target outputs:
install(TARGETS docopt_shared
        DESTINATION lib)
install(TARGETS docopt_static
        DESTINATION lib)

# Install the header files:
install(DIRECTORY ./ DESTINATION include/libdocopt
    FILES_MATCHING PATTERN "*.h")
