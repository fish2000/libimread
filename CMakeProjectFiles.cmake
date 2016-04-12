# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers

# To keep the file list clean
set(hdrs_dir ${${PROJECT_NAME}_include_dir})
set(srcs_dir ${CMAKE_CURRENT_SOURCE_DIR}/${source_dir})

SET(COMMON_LINK_FLAGS
    -m64 -mmacosx-version-min=10.9
    -fobjc-link-runtime)

# language-specific compile options
SET(C_OPTIONS
    -std=c99
    -Wno-incompatible-pointer-types
    -Wno-char-subscripts
    -x c)

SET(CXX_OPTIONS
    -std=c++1z -stdlib=libc++
    -x c++)

SET(OBJC_OPTIONS
    -fstack-protector
    -fobjc-abi-version=3
    -fno-objc-arc
    -fobjc-legacy-dispatch
    -std=c99 -ObjC
    -x objective-c)

SET(OBJCXX_OPTIONS
    -fstack-protector
    -fobjc-abi-version=3
    -fno-objc-arc
    -fobjc-legacy-dispatch
    -fobjc-call-cxx-cdtors
    -std=c++1z -stdlib=libc++
    -x objective-c++)

SET(OBJCXX_OPTIONS_ARC
    -fobjc-abi-version=3
    -fobjc-arc -fobjc-call-cxx-cdtors
    -fno-objc-arc-exceptions
    -std=c++1z -stdlib=libc++
    -x objective-c++)

# Extra options for Image IO code
SET(IO_EXTRA_OPTIONS
    -ffast-math)

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

# Project header files
set(hdrs
    ${PROJECT_BINARY_DIR}/libimread/libimread.hpp
    
    ${hdrs_dir}/ext/categories/NSBitmapImageRep+IM.hh
    ${hdrs_dir}/ext/categories/NSColor+IM.hh
    ${hdrs_dir}/ext/categories/NSData+IM.hh
    ${hdrs_dir}/ext/categories/NSDictionary+IM.hh
    ${hdrs_dir}/ext/categories/NSImage+CGImage.h
    ${hdrs_dir}/ext/categories/NSImage+QuickLook.h
    ${hdrs_dir}/ext/categories/NSImage+Resize.h
    ${hdrs_dir}/ext/categories/NSImage+ResizeBestFit.h
    ${hdrs_dir}/ext/categories/NSString+STL.hh
    ${hdrs_dir}/ext/categories/NSURL+IM.hh
    ${hdrs_dir}/ext/classes/AXCoreGraphicsImageRep.h
    ${hdrs_dir}/ext/classes/AXInterleavedImageRep.hh
    ${hdrs_dir}/ext/errors/backtrace.hh
    ${hdrs_dir}/ext/errors/demangle.hh
    ${hdrs_dir}/ext/errors/terminator.hh
    ${hdrs_dir}/ext/filesystem/directory.h
    ${hdrs_dir}/ext/filesystem/mode.h
    ${hdrs_dir}/ext/filesystem/opaques.h
    ${hdrs_dir}/ext/filesystem/path.h
    ${hdrs_dir}/ext/filesystem/resolver.h
    ${hdrs_dir}/ext/filesystem/temporary.h
    ${hdrs_dir}/ext/memory/fmemopen.hh
    ${hdrs_dir}/ext/memory/open_memstream.hh
    ${hdrs_dir}/ext/memory/refcount.hh
    ${hdrs_dir}/ext/JSON/json11.h
    ${hdrs_dir}/ext/butteraugli.hh
    ${hdrs_dir}/ext/exif.hh
    ${hdrs_dir}/ext/iod.hh
    ${hdrs_dir}/ext/pvr.hh
    ${hdrs_dir}/ext/pystring.hh
    ${hdrs_dir}/ext/WriteGIF.hh
    
    ${hdrs_dir}/IO/ansi.hh
    ${hdrs_dir}/IO/apple.hh
    ${hdrs_dir}/IO/bmp.hh
    ${hdrs_dir}/IO/gif.hh
    ${hdrs_dir}/IO/hdf5.hh
    ${hdrs_dir}/IO/jpeg.hh
    ${hdrs_dir}/IO/lsm.hh
    ${hdrs_dir}/IO/png.hh
    ${hdrs_dir}/IO/ppm.hh
    ${hdrs_dir}/IO/pvrtc.hh
    ${hdrs_dir}/IO/tiff.hh
    ${hdrs_dir}/IO/webp.hh
    # ${hdrs_dir}/IO/xcassets.hh
    
    ${hdrs_dir}/private/buffer_t.h
    ${hdrs_dir}/private/image_io.h
    ${hdrs_dir}/private/singleton.hh
    ${hdrs_dir}/private/static_image.h
    ${hdrs_dir}/private/vpp_symbols.hh
    
    # ${hdrs_dir}/process/jitresize.hh
    ${hdrs_dir}/process/neuquant.hh
    ${hdrs_dir}/process/neuquant.inl
    
    ${hdrs_dir}/objc-rt/objc-rt.hh
    ${hdrs_dir}/objc-rt/types.hh
    ${hdrs_dir}/objc-rt/selector.hh
    ${hdrs_dir}/objc-rt/message-args.hh
    ${hdrs_dir}/objc-rt/traits.hh
    ${hdrs_dir}/objc-rt/object.hh
    ${hdrs_dir}/objc-rt/message.hh
    ${hdrs_dir}/objc-rt/namespace-std.hh
    ${hdrs_dir}/objc-rt/namespace-im.hh
    ${hdrs_dir}/objc-rt/appkit.hh
    ${hdrs_dir}/objc-rt/appkit-pasteboard.hh
    
    ${hdrs_dir}/ansicolor.hh
    ${hdrs_dir}/base.hh
    ${hdrs_dir}/color.hh
    ${hdrs_dir}/coregraphics.hh
    ${hdrs_dir}/errors.hh
    ${hdrs_dir}/file.hh
    ${hdrs_dir}/formats.hh
    ${hdrs_dir}/fs.hh
    ${hdrs_dir}/halide.hh
    ${hdrs_dir}/hashing.hh
    ${hdrs_dir}/image.hh
    ${hdrs_dir}/imageformat.hh
    ${hdrs_dir}/interleaved.hh
    ${hdrs_dir}/memory.hh
    ${hdrs_dir}/options.hh
    ${hdrs_dir}/palette.hh
    ${hdrs_dir}/pixels.hh
    ${hdrs_dir}/preview.hh
    ${hdrs_dir}/rehash.hh
    ${hdrs_dir}/seekable.hh
    ${hdrs_dir}/symbols.hh
    ${IOD_SYMBOLS_HEADER}
    ${hdrs_dir}/traits.hh
)

set(preview_src ${srcs_dir}/plat/preview.cpp)
if(WINDOWS)
    set(preview_src ${srcs_dir}/plat/preview_windows.cpp)
endif(WINDOWS)
if(LINUX)
    set(preview_src ${srcs_dir}/plat/preview_linux.cpp)
endif(LINUX)
if(APPLE)
    set(preview_src ${srcs_dir}/plat/preview_mac.mm)
endif(APPLE)

# Project source files
set(srcs
    ${srcs_dir}/ext/categories/NSBitmapImageRep+IM.mm
    ${srcs_dir}/ext/categories/NSColor+IM.mm
    ${srcs_dir}/ext/categories/NSData+IM.mm
    ${srcs_dir}/ext/categories/NSDictionary+IM.mm
    ${srcs_dir}/ext/categories/NSImage+CGImage.m
    ${srcs_dir}/ext/categories/NSImage+QuickLook.m
    ${srcs_dir}/ext/categories/NSImage+Resize.m
    ${srcs_dir}/ext/categories/NSImage+ResizeBestFit.m
    ${srcs_dir}/ext/categories/NSString+STL.mm
    ${srcs_dir}/ext/categories/NSURL+IM.mm
    ${srcs_dir}/ext/classes/AXCoreGraphicsImageRep.m
    ${srcs_dir}/ext/classes/AXInterleavedImageRep.mm
    ${srcs_dir}/ext/errors/backtrace.cpp
    ${srcs_dir}/ext/errors/demangle.cpp
    ${srcs_dir}/ext/filesystem/opaques.cpp
    ${srcs_dir}/ext/filesystem/path.cpp
    ${srcs_dir}/ext/filesystem/temporary.cpp
    ${srcs_dir}/ext/memory/fmemopen.cpp
    ${srcs_dir}/ext/memory/open_memstream.cpp
    ${srcs_dir}/ext/memory/refcount.cpp
    ${srcs_dir}/ext/JSON/json11.cpp
    ${srcs_dir}/ext/JSON/schema.cpp
    ${srcs_dir}/ext/butteraugli.cpp
    ${srcs_dir}/ext/exif.cpp
    ${srcs_dir}/ext/pvr.cpp
    ${srcs_dir}/ext/pvrtc.cpp
    ${srcs_dir}/ext/pystring.cpp
    ${srcs_dir}/ext/WriteGIF.cpp
    
    ${srcs_dir}/IO/ansi.cpp
    ${srcs_dir}/IO/apple.mm
    ${srcs_dir}/IO/bmp.cpp
    ${srcs_dir}/IO/gif.cpp
    ${srcs_dir}/IO/hdf5.cpp
    ${srcs_dir}/IO/jpeg.cpp
    ${srcs_dir}/IO/lsm.cpp
    ${srcs_dir}/IO/lzw.cpp
    ${srcs_dir}/IO/png.cpp
    ${srcs_dir}/IO/ppm.cpp
    ${srcs_dir}/IO/pvrtc.cpp
    ${srcs_dir}/IO/tiff.cpp
    ${srcs_dir}/IO/webp.cpp
    # ${srcs_dir}/IO/xcassets.cpp
    
    # ${srcs_dir}/process/jitresize.cpp
    # ${srcs_dir}/process/neuquant.cpp
    
    ${srcs_dir}/objc-rt/appkit-pasteboard.mm
    ${srcs_dir}/objc-rt/namespace-std.mm
    ${srcs_dir}/objc-rt/selector.mm
    ${srcs_dir}/objc-rt/types.mm
    ${srcs_dir}/objc-rt/traits.mm
    
    ${srcs_dir}/ansicolor.cpp
    ${srcs_dir}/base.cpp
    ${srcs_dir}/color.cpp
    ${srcs_dir}/coregraphics.mm
    ${srcs_dir}/errors.cpp
    ${srcs_dir}/file.cpp
    ${srcs_dir}/formats.cpp
    ${srcs_dir}/fs.cpp
    ${srcs_dir}/halide.cpp
    ${srcs_dir}/hashing.cpp
    ${srcs_dir}/image.cpp
    ${srcs_dir}/imageformat.cpp
    ${srcs_dir}/interleaved.cpp
    ${srcs_dir}/memory.cpp
    ${srcs_dir}/options.cpp
    ${srcs_dir}/palette.cpp
    ${srcs_dir}/symbols.cpp
    ${preview_src}
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
    FIND_LIBRARY(SYSTEM_LIBRARY System)
    FIND_LIBRARY(COCOA_LIBRARY Cocoa)
    FIND_LIBRARY(FOUNDATION_LIBRARY Foundation)
    FIND_LIBRARY(COREFOUNDATION_LIBRARY CoreFoundation)
    FIND_LIBRARY(QUICKLOOK_LIBRARY QuickLook)
    
    MARK_AS_ADVANCED(SYSTEM_LIBRARY
                     COCOA_LIBRARY
                     FOUNDATION_LIBRARY
                     COREFOUNDATION_LIBRARY
                     QUICKLOOK_LIBRARY)
    
    SET(EXTRA_LIBS ${EXTRA_LIBS}
        ${SYSTEM_LIBRARY}
        ${COCOA_LIBRARY}
        ${FOUNDATION_LIBRARY}
        ${COREFOUNDATION_LIBRARY}
        ${QUICKLOOK_LIBRARY})
    
ENDIF(APPLE)

add_definitions(
    ${OBJCXX_OPTIONS}
    -Wno-nullability-completeness
    -DWITH_SCHEMA
    -O3 -mtune=native
    -fstrict-aliasing)
