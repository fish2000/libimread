
cmake_minimum_required(VERSION 3.0)
project(inicpp LANGUAGES C CXX)

set(SRC_DIR src)
set(INCLUDE_DIR include/inicpp)

# Compilation options:
add_definitions(
    -Wall -Werror
    -std=c++14 -stdlib=libc++ -O3
    -mtune=native -fstrict-aliasing)

set(SOURCE_FILES
    ${INCLUDE_DIR}/config.h
    ${SRC_DIR}/config.cpp
    ${INCLUDE_DIR}/exception.h
    ${INCLUDE_DIR}/option.h
    ${SRC_DIR}/option.cpp
    ${INCLUDE_DIR}/option_schema.h
    ${SRC_DIR}/option_schema.cpp
    ${INCLUDE_DIR}/parser.h
    ${SRC_DIR}/parser.cpp
    ${INCLUDE_DIR}/schema.h
    ${SRC_DIR}/schema.cpp
    ${INCLUDE_DIR}/section.h
    ${SRC_DIR}/section.cpp
    ${INCLUDE_DIR}/section_schema.h
    ${SRC_DIR}/section_schema.cpp
    ${INCLUDE_DIR}/types.h
    ${INCLUDE_DIR}/string_utils.h
    ${SRC_DIR}/string_utils.cpp
    ${INCLUDE_DIR}/inicpp.h
    ${INCLUDE_DIR}/dll.h
)

# Find header files in include directory
include_directories(include)

# Add an “OBJECT” library target called “inicpp”:
add_library(
    inicpp
    OBJECT ${SOURCE_FILES})
set_target_properties(
    inicpp
    PROPERTIES LIBRARY_OUTPUT_NAME "inicpp")
set_property(
    TARGET inicpp
    PROPERTY POSITION_INDEPENDENT_CODE 1)

# Add shared and static library targets, based on the “OBJECT” library,
# respectively  called “inicpp_shared” and “inicpp_static”:
add_library(
    inicpp_shared
    SHARED $<TARGET_OBJECTS:inicpp>)
add_library(
    inicpp_static
    STATIC $<TARGET_OBJECTS:inicpp>)

set_target_properties(
    inicpp_shared
    PROPERTIES LIBRARY_OUTPUT_NAME "inicpp")
set_target_properties(
    inicpp_shared
    PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(
    inicpp_static
    PROPERTIES ARCHIVE_OUTPUT_NAME "inicpp")
set_target_properties(
    inicpp_static
    PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(inicpp_shared)
target_link_libraries(inicpp_static)


# Set up google testing framework (not compile by default)
# set(gtest_force_shared_crt ON CACHE BOOL "Use shared (DLL) run-time lib even when Google Test is built as static lib.")
# add_subdirectory(vendor/googletest EXCLUDE_FROM_ALL)

# Add our very own unit tests (not compile by default)
# add_subdirectory(tests EXCLUDE_FROM_ALL)

# Add examples programs
# add_subdirectory(examples EXCLUDE_FROM_ALL)

# Install the header files:
install(DIRECTORY   include/inicpp/
        DESTINATION include/inicpp
        FILES_MATCHING PATTERN "*.h")

# Install the shared and static library target outputs:
install(TARGETS     inicpp_shared
        DESTINATION lib)
install(TARGETS     inicpp_static
        DESTINATION lib)
