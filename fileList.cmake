# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers

# To keep the file list clean
set(hdrs_dir ${${PROJECT_NAME}_include_dir})
set(srcs_dir ${CMAKE_CURRENT_SOURCE_DIR}/${source_dir})
# set(sdk_root "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk")

# common link flags
# SET(COMMON_LINK_FLAGS
#     -m64 -mmacosx-version-min=10.9
#     -fobjc-link-runtime
#     -rpath @executable_path/../Frameworks)
# -isysroot ${sdk_root}
# -syslibroot ${sdk_root}
SET(COMMON_LINK_FLAGS
    -m64 -mmacosx-version-min=10.9
    -fobjc-link-runtime)

# language-specific compile options
SET(C_OPTIONS
    -std=c99 -Wno-incompatible-pointer-types -Wno-char-subscripts
    -x c)

SET(CXX_OPTIONS
    -std=c++14 -stdlib=libc++
    -x c++)

SET(OBJC_OPTIONS
    -fobjc-link-runtime
    -fobjc-abi-version=3 -fobjc-arc -std=c99
    -ObjC
    -x objective-c)

# -ObjC++
SET(OBJCXX_OPTIONS
    -fobjc-abi-version=3 -fobjc-arc -fobjc-call-cxx-cdtors
    -std=c++14 -stdlib=libc++
    -x objective-c++)

# Extra options for Image IO code
SET(IO_EXTRA_OPTIONS
    -ffast-math)

# SET(C_FILES "" PARENT_SCOPE)
# SET(CC_FILES "" PARENT_SCOPE)
# SET(M_FILES "" PARENT_SCOPE)
# SET(MM_FILES "" PARENT_SCOPE)
# SET(IO_FILES "" PARENT_SCOPE)
SET(C_FILES "")
SET(CC_FILES "")
SET(M_FILES "")
SET(MM_FILES "")
SET(IO_FILES "")

FILE(
    GLOB_RECURSE C_FILES
    "${srcs_dir}/*.c")
FILE(
    GLOB_RECURSE CC_FILES
    "${srcs_dir}/*.cpp")
FILE(
    GLOB_RECURSE M_FILES
    "${srcs_dir}/*.m")
FILE(
    GLOB_RECURSE MM_FILES
    "${srcs_dir}/*.mm")
FILE(
    GLOB_RECURSE IO_FILES
    "${srcs_dir}/IO/*.*")

SEPARATE_ARGUMENTS(COMMON_LINK_FLAGS)

SEPARATE_ARGUMENTS(C_FILES)
SEPARATE_ARGUMENTS(CC_FILES)
SEPARATE_ARGUMENTS(M_FILES)
SEPARATE_ARGUMENTS(MM_FILES)
SEPARATE_ARGUMENTS(IO_FILES)

SEPARATE_ARGUMENTS(C_OPTIONS)
SEPARATE_ARGUMENTS(CXX_OPTIONS)
SEPARATE_ARGUMENTS(OBJC_OPTIONS)
SEPARATE_ARGUMENTS(OBJCXX_OPTIONS)
SEPARATE_ARGUMENTS(IO_EXTRA_OPTIONS)

# Configure the project-settings header file
configure_file(
    "${hdrs_dir}/libimread.hpp.in"
    "${PROJECT_BINARY_DIR}/libimread/libimread.hpp")

# Project header files
set(hdrs
    ${PROJECT_BINARY_DIR}/libimread/libimread.hpp
    # ${PROJECT_BINARY_DIR}/libimread/symbols.hpp
    
    ${hdrs_dir}/ext/filesystem/path.h
    ${hdrs_dir}/ext/filesystem/resolver.h
    ${hdrs_dir}/ext/JSON/json11.h
    ${hdrs_dir}/ext/fmemopen.hh
    ${hdrs_dir}/ext/open_memstream.hh
    ${hdrs_dir}/ext/pvr.h
    # ${hdrs_dir}/ext/UTI.h
    ${hdrs_dir}/ext/WriteGIF.h
    
    ${hdrs_dir}/IO/apple.hh
    ${hdrs_dir}/IO/bmp.hh
    ${hdrs_dir}/IO/gif.hh
    ${hdrs_dir}/IO/jpeg.hh
    ${hdrs_dir}/IO/lsm.hh
    ${hdrs_dir}/IO/png.hh
    ${hdrs_dir}/IO/ppm.hh
    ${hdrs_dir}/IO/pvrtc.hh
    ${hdrs_dir}/IO/tiff.hh
    ${hdrs_dir}/IO/webp.hh
    ${hdrs_dir}/IO/xcassets.hh
    
    ${hdrs_dir}/private/buffer_t.h
    ${hdrs_dir}/private/image_io.h
    ${hdrs_dir}/private/spx_defines.h
    ${hdrs_dir}/private/static_image.h
    ${hdrs_dir}/private/vpp_symbols.hh
    
    ${hdrs_dir}/process/jitresize.hh
    ${hdrs_dir}/process/neuquant.h
    ${hdrs_dir}/process/neuquant.inl
    
    ${hdrs_dir}/ansicolor.hh
    ${hdrs_dir}/base.hh
    ${hdrs_dir}/coregraphics.hh
    ${hdrs_dir}/errors.hh
    ${hdrs_dir}/file.hh
    ${hdrs_dir}/formats.hh
    ${hdrs_dir}/fs.hh
    ${hdrs_dir}/halide.hh
    ${hdrs_dir}/image.hh
    ${hdrs_dir}/imageformat.hh
    ${hdrs_dir}/memory.hh
    ${hdrs_dir}/objc-rt.hh
    ${hdrs_dir}/options.hh
    ${hdrs_dir}/pixels.hh
    ${hdrs_dir}/seekable.hh
    ${hdrs_dir}/symbols.hh
    ${IOD_SYMBOLS_HEADER}
    ${hdrs_dir}/tools.hh
    ${hdrs_dir}/traits.hh
    # ${hdrs_dir}/vpp.hh
)

# Project source files
set(srcs
    ${srcs_dir}/ext/filesystem/path.cpp
    ${srcs_dir}/ext/JSON/json11.cpp
    ${srcs_dir}/ext/JSON/schema.cpp
    ${srcs_dir}/ext/fmemopen.cpp
    ${srcs_dir}/ext/open_memstream.cpp
    ${srcs_dir}/ext/pvr.cpp
    ${srcs_dir}/ext/pvrtc.cpp
    # ${srcs_dir}/ext/UTI.mm
    ${srcs_dir}/ext/WriteGIF.cpp
    
    ${srcs_dir}/IO/apple.mm
    ${srcs_dir}/IO/bmp.cpp
    ${srcs_dir}/IO/gif.cpp
    ${srcs_dir}/IO/jpeg.cpp
    ${srcs_dir}/IO/lsm.cpp
    ${srcs_dir}/IO/lzw.cpp
    ${srcs_dir}/IO/png.cpp
    ${srcs_dir}/IO/ppm.cpp
    ${srcs_dir}/IO/pvrtc.cpp
    ${srcs_dir}/IO/tiff.cpp
    ${srcs_dir}/IO/webp.cpp
    ${srcs_dir}/IO/xcassets.cpp
    
    ${srcs_dir}/process/jitresize.cpp
    ${srcs_dir}/process/neuquant.cpp
    
    ${srcs_dir}/ansicolor.cpp
    ${srcs_dir}/base.cpp
    ${srcs_dir}/coregraphics.mm
    ${srcs_dir}/errors.cpp
    ${srcs_dir}/file.cpp
    ${srcs_dir}/formats.cpp
    ${srcs_dir}/fs.cpp
    ${srcs_dir}/halide.cpp
    ${srcs_dir}/options.cpp
    ${srcs_dir}/symbols.cpp
    # ${srcs_dir}/vpp.cpp
)

# set common link flags
foreach(FLAG IN LISTS ${COMMON_LINK_FLAGS})
    
    set_source_files_properties(
        ${srcs}
        PROPERTIES LINK_FLAGS ${LINK_FLAGS} ${FLAG})
    
endforeach()

# set source properties,
# using /((?P<objc>(ObjC|Objective-C))|(?P<c>(C)))?(?P<plus>(C|\+\+|PP|XX))?/ language-specific stuff
foreach(src_file IN LISTS ${srcs})
    if(${src_file} MATCHES "\.c$")
        set(opts ${C_OPTIONS})
    elseif(${src_file} MATCHES "\.cpp$")
        set(opts ${CXX_OPTIONS})
    elseif(${src_file} MATCHES "\.m$")
        set(opts ${OBJC_OPTIONS})
    elseif(${src_file} MATCHES "\.mm$")
        set(opts ${OBJCXX_OPTIONS})
    else()
        set(opts "")
    endif()
    separate_arguments(${opts})
    if(${src_file} MATCHES "(.*)IO/(.*)")
        # Extra source file props specific to Image IO code compilation
        foreach(extra_opt IN LISTS ${IO_EXTRA_OPTIONS})
            list(APPEND opts ${extra_opt})
        endforeach()
    endif()
    separate_arguments(${opts})
    foreach(opt IN LISTS ${opts})
        get_source_file_property(existant_compile_flags ${src_file} COMPILE_FLAGS)
        set_source_files_properties(${src_file}
            PROPERTIES COMPILE_FLAGS ${existant_compile_flags} ${opt})
    endforeach()
endforeach()

IF(APPLE)
    # Right now there are no non-Apple options that work
    # INCLUDE_DIRECTORIES(/Developer/Headers/FlatCarbon)
    FIND_LIBRARY(COCOA_LIBRARY Cocoa)
    FIND_LIBRARY(FOUNDATION_LIBRARY Foundation)
    FIND_LIBRARY(COREFOUNDATION_LIBRARY CoreFoundation)
    MARK_AS_ADVANCED(COCOA_LIBRARY
                     FOUNDATION_LIBRARY
                     COREFOUNDATION_LIBRARY)
    SET(EXTRA_LIBS
        ${EXTRA_LIBS}
        ${COCOA_LIBRARY} ${FOUNDATION_LIBRARY}
        ${COREFOUNDATION_LIBRARY})
    
    # set_source_files_properties(GLOB_RECURSE "${srcs_dir}/*.mm"
    #     PROPERTIES COMPILE_FLAGS ${COMPILE_FLAGS} -ObjC++)
    # set_source_files_properties(GLOB_RECURSE "${srcs_dir}/*.mm"
    #     PROPERTIES COMPILE_FLAGS ${COMPILE_FLAGS} -fobjc-arc)
    
ENDIF(APPLE)

add_definitions(
    ${OBJCXX_OPTIONS}
    -DWITH_SCHEMA
    -O3 -mtune=native -fstrict-aliasing)

