# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers
cmake_minimum_required(VERSION 3.0)
project(libguid LANGUAGES C CXX)

# TODO: compile examples and tests (why not??)
# "testmain.cpp"
# "test.cpp"

set(libguid_VERSION_MAJOR "0")
set(libguid_VERSION_MINOR "3")
set(libguid_VERSION_PATCH "5")

add_definitions(
    -Wall -Werror
    -std=c++17 -stdlib=libc++ -O3
    -mtune=native -fstrict-aliasing)

if(APPLE)
    find_library(COREFOUNDATION_LIBRARY CoreFoundation)
    add_definitions(-DGUID_CFUUID)
endif(APPLE)

set(COMMON_LINK_FLAGS
    -m64)

set(srcs
    "guid.cpp")

foreach(FLAG IN LISTS ${COMMON_LINK_FLAGS})

    set_source_files_properties(
        ${srcs}
        PROPERTIES LINK_FLAGS ${LINK_FLAGS} ${FLAG})

endforeach()

set(hdrs
    "guid.h")

# Add an “OBJECT” library target called “guid”:
add_library(
    guid
    OBJECT ${srcs} ${hdrs})
set_target_properties(
    guid
    PROPERTIES LIBRARY_OUTPUT_NAME "guid")
set_property(
    TARGET guid
    PROPERTY POSITION_INDEPENDENT_CODE 1)

# Add shared and static library targets, based on the “OBJECT” library,
# respectively  called “guid_shared” and “guid_static”:
add_library(
    guid_shared
    SHARED $<TARGET_OBJECTS:guid>)
add_library(
    guid_static
    STATIC $<TARGET_OBJECTS:guid>)
set_target_properties(
    guid_shared
    PROPERTIES LIBRARY_OUTPUT_NAME "guid")
set_target_properties(
    guid_static
    PROPERTIES ARCHIVE_OUTPUT_NAME "guid")
target_link_libraries(guid_shared ${COREFOUNDATION_LIBRARY})
target_link_libraries(guid_static ${COREFOUNDATION_LIBRARY})

# Install the shared and static library target outputs:
install(TARGETS guid_shared DESTINATION lib)
install(TARGETS guid_static DESTINATION lib)

# Install the header files:
install(DIRECTORY ./ DESTINATION include/libguid
    FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ./ DESTINATION include/libguid
    FILES_MATCHING PATTERN "*.hh")
