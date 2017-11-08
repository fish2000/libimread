cmake_minimum_required(VERSION 3.7)
project(SG14 LANGUAGES CXX)

add_library(SG14 INTERFACE)

# determine whether this is a standalone project or included by other projects
set(SG14_STANDALONE_PROJECT OFF)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(SG14_STANDALONE_PROJECT ON)
endif()

if(CMAKE_MAJOR_VERSION VERSION_LESS 3.7.9)
    if(NOT MSVC)
        include(CheckCXXCompilerFlag)
        CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
        CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
        if(COMPILER_SUPPORTS_CXX17)
            target_compile_options(SG14 INTERFACE "-std=c++17")
        elseif(COMPILER_SUPPORTS_CXX14)
            target_compile_options(SG14 INTERFACE "-std=c++14")
        else()
            message(FATAL_ERROR "${CMAKE_CXX_COMPILER} has neither C++17 nor C++14 support")
        endif()
    endif()
else()
    # set the GSL library to be compiled only with c++14
    target_compile_features(SG14 INTERFACE cxx_std_14)
    # on *nix systems force the use of -std=c++XX instead of -std=gnu++XX (default)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# add definitions to the library and targets that consume it
target_compile_definitions(SG14 INTERFACE
    # remove unnecessary warnings about unchecked iterators
    $<$<CXX_COMPILER_ID:MSVC>:_SCL_SECURE_NO_WARNINGS>)

# add include folders to the library and targets that consume it
target_include_directories(SG14 INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.hpp")
