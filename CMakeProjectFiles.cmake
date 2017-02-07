# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers

# To keep the file list clean
set(hdrs_dir ${${PROJECT_NAME}_include_dir})
set(srcs_dir ${CMAKE_CURRENT_SOURCE_DIR}/${source_dir})

# language-specific compile options
SET(C_OPTIONS
    -std=c99
    -Wno-incompatible-pointer-types
    -Wno-char-subscripts
    -x c)

SET(CXX_OPTIONS
    -std=c++1z -stdlib=libc++
    -x c++)

# Extra options for Image IO code
SET(IO_EXTRA_OPTIONS
    -ffast-math)

SET(C_FILES "")
SET(CC_FILES "")
SET(IO_FILES "")

FILE(
    GLOB_RECURSE C_FILES
    "${srcs_dir}/*.c")
FILE(
    GLOB_RECURSE CC_FILES
    "${srcs_dir}/*.cpp")
FILE(
    GLOB_RECURSE IO_FILES
    "${srcs_dir}/IO/*.*")

SEPARATE_ARGUMENTS(C_FILES)
SEPARATE_ARGUMENTS(CC_FILES)
SEPARATE_ARGUMENTS(IO_FILES)

SEPARATE_ARGUMENTS(C_OPTIONS)
SEPARATE_ARGUMENTS(CXX_OPTIONS)
SEPARATE_ARGUMENTS(IO_EXTRA_OPTIONS)

# Project header files
set(hdrs
    ${CAPN_DIR}/halide.capnp.h
    ${CAPN_DIR}/numpy.capnp.h
    ${PROJECT_BINARY_DIR}/libimread/libimread.hpp
    ${hdrs_dir}/ext/errors/backtrace.hh
    ${hdrs_dir}/ext/errors/demangle.hh
    ${hdrs_dir}/ext/errors/terminator.hh
    ${hdrs_dir}/ext/filesystem/directory.h
    ${hdrs_dir}/ext/filesystem/mode.h
    ${hdrs_dir}/ext/filesystem/nowait.h
    ${hdrs_dir}/ext/filesystem/opaques.h
    ${hdrs_dir}/ext/filesystem/path.h
    ${hdrs_dir}/ext/filesystem/resolver.h
    ${hdrs_dir}/ext/filesystem/temporary.h
    ${hdrs_dir}/ext/memory/fmemopen.hh
    ${hdrs_dir}/ext/memory/open_memstream.hh
    # ${hdrs_dir}/ext/memory/refcount.hh
    ${hdrs_dir}/ext/JSON/json11.h
    ${hdrs_dir}/ext/base64.hh
    ${hdrs_dir}/ext/butteraugli.hh
    ${hdrs_dir}/ext/exif.hh
    ${hdrs_dir}/ext/lzw.hh
    ${hdrs_dir}/ext/MurmurHash2.hh
    ${hdrs_dir}/ext/iod.hh
    ${hdrs_dir}/ext/pvr.hh
    ${hdrs_dir}/ext/pystring.hh
    ${hdrs_dir}/ext/valarray.hh
    ${hdrs_dir}/ext/WriteGIF.hh

    ${hdrs_dir}/IO/ansi.hh
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

    ${hdrs_dir}/private/buffer_t.h
    ${hdrs_dir}/private/image_io.h
    # ${hdrs_dir}/private/singleton.hh
    ${hdrs_dir}/private/static_image.h
    # ${hdrs_dir}/private/vpp_symbols.hh

    # ${hdrs_dir}/process/jitresize.hh
    ${hdrs_dir}/process/neuquant.hh
    ${hdrs_dir}/process/neuquant.inl

    ${hdrs_dir}/ansicolor.hh
    ${hdrs_dir}/base.hh
    ${hdrs_dir}/color.hh
    ${hdrs_dir}/errors.hh
    ${hdrs_dir}/file.hh
    ${hdrs_dir}/filehandle.hh
    ${hdrs_dir}/formats.hh
    ${hdrs_dir}/gzio.hh
    ${hdrs_dir}/halide.hh
    ${hdrs_dir}/hashing.hh
    ${hdrs_dir}/histogram.hh
    ${hdrs_dir}/image.hh
    ${hdrs_dir}/imageformat.hh
    ${hdrs_dir}/imagelist.hh
    ${hdrs_dir}/imageview.hh
    ${hdrs_dir}/interleaved.hh
    ${hdrs_dir}/iterators.hh
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
    ${CAPN_DIR}/halide.capnp.c++
    ${CAPN_DIR}/numpy.capnp.c++
    ${srcs_dir}/ext/errors/backtrace.cpp
    ${srcs_dir}/ext/errors/demangle.cpp
    ${srcs_dir}/ext/filesystem/nowait.cpp
    ${srcs_dir}/ext/filesystem/opaques.cpp
    ${srcs_dir}/ext/filesystem/path.cpp
    ${srcs_dir}/ext/filesystem/temporary.cpp
    ${srcs_dir}/ext/memory/fmemopen.cpp
    ${srcs_dir}/ext/memory/open_memstream.cpp
    # ${srcs_dir}/ext/memory/refcount.cpp
    ${srcs_dir}/ext/JSON/json11.cpp
    ${srcs_dir}/ext/JSON/schema.cpp
    ${srcs_dir}/ext/base64.cpp
    ${srcs_dir}/ext/butteraugli.cpp
    ${srcs_dir}/ext/exif.cpp
    ${srcs_dir}/ext/lzw.cpp
    ${srcs_dir}/ext/MurmurHash2.cpp
    ${srcs_dir}/ext/pvr.cpp
    ${srcs_dir}/ext/pvrtc.cpp
    ${srcs_dir}/ext/pystring.cpp
    ${srcs_dir}/ext/WriteGIF.cpp

    # ${srcs_dir}/IO/ansi.cpp
    ${srcs_dir}/IO/bmp.cpp
    ${srcs_dir}/IO/gif.cpp
    ${srcs_dir}/IO/hdf5.cpp
    ${srcs_dir}/IO/jpeg.cpp
    ${srcs_dir}/IO/lsm.cpp
    ${srcs_dir}/IO/png.cpp
    ${srcs_dir}/IO/ppm.cpp
    ${srcs_dir}/IO/pvrtc.cpp
    ${srcs_dir}/IO/tiff.cpp
    ${srcs_dir}/IO/webp.cpp

    # ${srcs_dir}/process/jitresize.cpp
    # ${srcs_dir}/process/neuquant.cpp

    # ${srcs_dir}/ansicolor.cpp
    # ${srcs_dir}/base.cpp
    # ${srcs_dir}/color.cpp
    ${srcs_dir}/errors.cpp
    ${srcs_dir}/file.cpp
    ${srcs_dir}/filehandle.cpp
    ${srcs_dir}/formats.cpp
    ${srcs_dir}/gzio.cpp
    # ${srcs_dir}/halide.cpp
    ${srcs_dir}/hashing.cpp
    ${srcs_dir}/histogram.cpp
    ${srcs_dir}/image.cpp
    ${srcs_dir}/imageformat.cpp
    ${srcs_dir}/imagelist.cpp
    ${srcs_dir}/imageview.cpp
    # ${srcs_dir}/interleaved.cpp
    ${srcs_dir}/iterators.cpp
    ${srcs_dir}/memory.cpp
    ${srcs_dir}/options.cpp
    # ${srcs_dir}/palette.cpp
    # ${srcs_dir}/pixels.cpp
    ${srcs_dir}/seekable.cpp
    # ${srcs_dir}/symbols.cpp
    ${preview_src}
)

# set source properties
foreach(src_file IN LISTS ${srcs})
    if(${src_file} MATCHES "\.c$")
        set(opts ${C_OPTIONS})
    elseif(${src_file} MATCHES "\.cpp$")
        set(opts ${CXX_OPTIONS})
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

add_definitions(
    ${CXX_OPTIONS}
    -DWITH_SCHEMA
    -O3 -funroll-loops
    -mtune=native
    -fstrict-aliasing)
