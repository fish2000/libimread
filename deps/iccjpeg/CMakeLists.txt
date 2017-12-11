# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers
cmake_minimum_required(VERSION 3.5)
project(iccjpeg LANGUAGES C)

set(iccjpeg_VERSION_MAJOR "0")
set(iccjpeg_VERSION_MINOR "3")
set(iccjpeg_VERSION_PATCH "5")

# Compilation options:
add_definitions(
    -DIS_LITTLE_ENDIAN
    -Wall -Werror
    -std=c99 -O3
    -Wno-incompatible-pointer-types
    -mtune=native -fstrict-aliasing
    -x c)

# Require jpeglib:
find_package(JPEG REQUIRED)

# Include and link directories:
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

# Project header file:
set(hdrs "iccjpeg/iccjpeg.h")

# Project source file:
set(srcs "iccjpeg/iccjpeg.c")

# Add an “OBJECT” library target called “iccjpeg”:
add_library(
    iccjpeg
    OBJECT ${srcs} ${hdrs})
set_target_properties(
    iccjpeg
    PROPERTIES LIBRARY_OUTPUT_NAME "iccjpeg")
set_property(
    TARGET iccjpeg
    PROPERTY POSITION_INDEPENDENT_CODE 1)

# Add shared and static library targets, based on the “OBJECT” library,
# respectively called “iccjpeg_shared” and “iccjpeg_static”:
add_library(
    iccjpeg_shared
    SHARED $<TARGET_OBJECTS:iccjpeg>)
add_library(
    iccjpeg_static
    STATIC $<TARGET_OBJECTS:iccjpeg>)
set_target_properties(
    iccjpeg_shared
    PROPERTIES LIBRARY_OUTPUT_NAME "iccjpeg")
set_target_properties(
    iccjpeg_static
    PROPERTIES ARCHIVE_OUTPUT_NAME "iccjpeg")
target_link_libraries(iccjpeg_shared ${JPEG_LIBRARIES})
target_link_libraries(iccjpeg_static ${JPEG_LIBRARIES})

# Install the shared and static library target outputs:
install(TARGETS iccjpeg_shared
        DESTINATION lib)
install(TARGETS iccjpeg_static
        DESTINATION lib)

# Install the header files:
install(DIRECTORY   iccjpeg/
        DESTINATION include/iccjpeg
        FILES_MATCHING PATTERN "*.h")
