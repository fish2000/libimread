cmake_minimum_required(VERSION 3.8.0)

project(GSL CXX)

# include(ExternalProject)
# find_package(Git)

# creates a library GSL which is an interface (header files only)
add_library(GSL INTERFACE)

# determine whether this is a standalone project or included by other projects
set(GSL_STANDALONE_PROJECT OFF)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(GSL_STANDALONE_PROJECT ON)
endif()

# when minimum version required is 3.8.0 remove if below
# both branches do exactly the same thing
if(CMAKE_MAJOR_VERSION VERSION_LESS 3.7.9)
    if(NOT MSVC)
        include(CheckCXXCompilerFlag)
        CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
        CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
        if(COMPILER_SUPPORTS_CXX17)
            target_compile_options(GSL INTERFACE "-std=c++17")
        elseif(COMPILER_SUPPORTS_CXX14)
            target_compile_options(GSL INTERFACE "-std=c++14")
        else()
            message(FATAL_ERROR "${CMAKE_CXX_COMPILER} has neither C++17 nor C++14 support")
        endif()
    endif()
else()
    # set the GSL library to be compiled only with c++14
    # target_compile_features(GSL INTERFACE cxx_std_14)
    target_compile_features(GSL INTERFACE cxx_std_17)
    # on *nix systems force the use of -std=c++XX instead of -std=gnu++XX (default)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# add definitions to the library and targets that consume it
target_compile_definitions(GSL INTERFACE
    # remove unnecessary warnings about unchecked iterators
    $<$<CXX_COMPILER_ID:MSVC>:_SCL_SECURE_NO_WARNINGS>)

# add include folders to the library and targets that consume it
target_include_directories(GSL INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# add natvis file to the library so it will automatically be loaded into Visual Studio
# target_sources(GSL INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/GSL.natvis)

install(DIRECTORY include/gsl
        DESTINATION include)

# option(GSL_TEST "Generate tests." ${GSL_STANDALONE_PROJECT})
# if(GSL_TEST)
#     enable_testing()
#     add_subdirectory(tests)
# endif()
