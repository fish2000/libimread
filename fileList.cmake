# Author: Félix C. Morency
# 2011.10

#To keep the file list clean
set(hdrs_dir ${${PROJECT_NAME}_include_dir})
set(srcs_dir ${CMAKE_CURRENT_SOURCE_DIR}/${source_dir})

#Project header files
set(hdrs
    ${hdrs_dir}/private/buffer_t.h
    ${hdrs_dir}/_bmp.h
    ${hdrs_dir}/_jpeg.h
    ${hdrs_dir}/_lsm.h
    ${hdrs_dir}/_png.h
    ${hdrs_dir}/_pvrtc.h
    ${hdrs_dir}/_tiff.h
    ${hdrs_dir}/_webp.h
    ${hdrs_dir}/base.h
    ${hdrs_dir}/errors.h
    ${hdrs_dir}/file.h
    ${hdrs_dir}/formats.h
    ${hdrs_dir}/memory.h
    # ${hdrs_dir}/numpy.h
    ${hdrs_dir}/halide.h
    ${hdrs_dir}/pvr.h
    ${hdrs_dir}/tools.h
)

#Project source files
set(srcs    
    ${srcs_dir}/_bmp.cpp
    ${srcs_dir}/_jpeg.cpp
    ${srcs_dir}/_lsm.cpp
    ${srcs_dir}/_png.cpp
    ${srcs_dir}/_pvrtc.cpp
    ${srcs_dir}/_tiff.cpp
    ${srcs_dir}/_webp.cpp
    ${srcs_dir}/formats.cpp
    ${srcs_dir}/lzw.cpp
    # ${srcs_dir}/numpy.cpp
    ${srcs_dir}/halide.cpp
    ${srcs_dir}/pvr.cpp
    ${srcs_dir}/pvrtc.cpp
)
