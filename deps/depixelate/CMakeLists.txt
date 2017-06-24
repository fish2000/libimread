# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers
cmake_minimum_required(VERSION 3.0)
project(depixelate LANGUAGES C CXX)

set(depixelate_VERSION_MAJOR "0")
set(depixelate_VERSION_MINOR "3")
set(depixelate_VERSION_PATCH "0")

add_subdirectory(hqx)
add_subdirectory(scale2x)
add_subdirectory(xbrz)

find_package(PkgConfig REQUIRED)
pkg_search_module(IL REQUIRED IL)
find_library(FOUNDATION_LIBRARY Foundation)
find_library(APPKIT_LIBRARY AppKit)

include_directories(${IL_INCLUDE_DIRS})

add_definitions(
    -Wall -Werror -fstack-protector-all
    -fobjc-arc -fobjc-abi-version=3
    -fobjc-call-cxx-cdtors
    -fobjc-arc-exceptions
    -std=gnu++0x -stdlib=libc++ -O3
    -mtune=native -fstrict-aliasing
    -x objective-c++)

set(srcs "main.mm")

# Add an “OBJECT” library target called “depixelate”:
add_library(
    depixelate
    OBJECT ${srcs})
set_target_properties(
    depixelate
    PROPERTIES LIBRARY_OUTPUT_NAME "depixelate")
set_property(
    TARGET depixelate
    PROPERTY POSITION_INDEPENDENT_CODE 1)
add_dependencies(depixelate depixelate_hqx)
add_dependencies(depixelate depixelate_scale2x)
add_dependencies(depixelate depixelate_xbrz)

# Add shared and static library targets, based on the “OBJECT” library,
# respectively called “depixelate_shared” and “depixelate_static”:
add_library(
    depixelate_shared
    SHARED $<TARGET_OBJECTS:depixelate>
           $<TARGET_OBJECTS:depixelate_hqx>
           $<TARGET_OBJECTS:depixelate_scale2x>
           $<TARGET_OBJECTS:depixelate_xbrz>)
add_library(
    depixelate_static
    STATIC $<TARGET_OBJECTS:depixelate>
           $<TARGET_OBJECTS:depixelate_hqx>
           $<TARGET_OBJECTS:depixelate_scale2x>
           $<TARGET_OBJECTS:depixelate_xbrz>)
set_target_properties(
    depixelate_shared
    PROPERTIES LIBRARY_OUTPUT_NAME "depixelate")
set_target_properties(
    depixelate_static
    PROPERTIES ARCHIVE_OUTPUT_NAME "depixelate")

target_link_libraries(depixelate_shared ${IL_LIBRARIES}
                                        ${FOUNDATION_LIBRARY}
                                        ${APPKIT_LIBRARY})

target_link_libraries(depixelate_static ${IL_LIBRARIES}
                                        ${FOUNDATION_LIBRARY}
                                        ${APPKIT_LIBRARY})

# Install the shared and static library target outputs:
install(TARGETS depixelate_shared
        DESTINATION lib)
install(TARGETS depixelate_static
        DESTINATION lib)

# Install the header files:
# install(DIRECTORY ./ DESTINATION include/depixelate
#     FILES_MATCHING PATTERN "*.h")
