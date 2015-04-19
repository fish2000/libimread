# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers

# To keep the file list clean
set(hdrs_dir ${${PROJECT_NAME}_include_dir})
set(srcs_dir ${CMAKE_CURRENT_SOURCE_DIR}/${source_dir})

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
    ${hdrs_dir}/ext/fmemopen.hh
    ${hdrs_dir}/ext/open_memstream.hh
    ${hdrs_dir}/ext/pvr.h
    ${hdrs_dir}/ext/UTI.h
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
    ${hdrs_dir}/options.hh
    ${hdrs_dir}/pixels.hh
    ${hdrs_dir}/seekable.hh
    ${hdrs_dir}/symbols.hh
    ${hdrs_dir}/tools.hh
    ${hdrs_dir}/traits.hh
    # ${hdrs_dir}/vpp.hh
)

# Project source files
set(srcs
    ${srcs_dir}/ext/filesystem/path.cpp
    ${srcs_dir}/ext/fmemopen.cpp
    ${srcs_dir}/ext/open_memstream.cpp
    ${srcs_dir}/ext/pvr.cpp
    ${srcs_dir}/ext/pvrtc.cpp
    ${srcs_dir}/ext/UTI.mm
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
    ${srcs_dir}/file.cpp
    ${srcs_dir}/formats.cpp
    ${srcs_dir}/fs.cpp
    ${srcs_dir}/halide.cpp
    ${srcs_dir}/options.cpp
    ${srcs_dir}/symbols.cpp
    # ${srcs_dir}/vpp.cpp
)
