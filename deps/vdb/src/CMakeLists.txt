# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers
cmake_minimum_required(VERSION 3.3)
project(imagecompression LANGUAGES C CXX)

set(imagecompression_VERSION_MAJOR "0")
set(imagecompression_VERSION_MINOR "1")
set(imagecompression_VERSION_PATCH "0")

find_package(FLTK REQUIRED)
find_package(zlib REQUIRED)

if(APPLE)
    find_library(CARBON_FMWK Carbon)
    find_library(COCOA_FMWK Cocoa)
    find_library(AGL_FMWK AGL)
    find_library(OPENGL_FMWK OpenGL)
    find_library(APPLICATIONSERVICES_FMWK ApplicationServices)
    mark_as_advanced(CARBON_FMWK
                     COCOA_FMWK
                     AGL_FMWK
                     OPENGL_FMWK
                     APPLICATIONSERVICES_FMWK)
    set(FMWKS ${CARBON_FMWK}
              ${COCOA_FMWK}
              ${AGL_FMWK}
              ${OPENGL_FMWK}
              ${APPLICATIONSERVICES_FMWK})
    set(FLTK_CXX_FLAGS)
    set(FLTK_LD_FLAGS)
else(APPLE)
    find_package(OpenGL REQUIRED)
    execute_process(COMMAND "`which fltk-config` --use-gl --cxxflags"
                    OUTPUT_VARIABLE FLTK_CXX_FLAGS)
    execute_process(COMMAND "`which fltk-config` --use-gl --ldstaticflags"
                    OUTPUT_VARIABLE FLTK_LD_FLAGS)
endif(APPLE)


add_definitions(
    -Wall -Werror
    -Wno-deprecated-declarations
    -std=c++14 -stdlib=libc++ -O3
    -mtune=native -fstrict-aliasing
    ${FLTK_CXX_FLAGS})

set(COMMON_LINK_FLAGS
    -m64
    ${FLTK_LD_FLAGS})

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${FLTK_INCLUDE_DIR})

set(hdrs
    "Frame.h"
    "glext.h"
    "GLWindow.h"
    "LabelTable.h"
    "SocketManager.h"
    "StringTable.h"
    "trackball.h"
    "VDBWindow.h"
    "wglext.h")

set(srcs
    "main.cpp"
    "VDBWindow.cpp"
    "SocketManager.cpp"
    "Frame.cpp"
    "trackball.cpp"
    "GLWindow.cpp")

foreach(FLAG IN LISTS ${COMMON_LINK_FLAGS})

    set_source_files_properties(
        ${srcs}
        PROPERTIES LINK_FLAGS ${LINK_FLAGS} ${FLAG})

endforeach()

add_executable(vdb ${srcs} ${hdrs})

if(APPLE)

    target_link_libraries(vdb
        ${FMWKS}
        ${FLTK_LIBRARIES})

else(APPLE)

    target_link_libraries(vdb
        ${FLTK_LIBRARIES})

endif(APPLE)

install(TARGETS vdb DESTINATION lib)
# install(DIRECTORY ./base/ DESTINATION include/imagecompression/base
#     FILES_MATCHING PATTERN "*.h")
# install(DIRECTORY ./image_compression/internal/ DESTINATION include/imagecompression/internal
#     FILES_MATCHING PATTERN "*.h")
# install(DIRECTORY ./image_compression/public/ DESTINATION include/imagecompression/public
#     FILES_MATCHING PATTERN "*.h")
