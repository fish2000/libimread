# Author: Alexander Böhn
# © 2011.10 -- GPL, Motherfuckers

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

set(TEST_SOURCES ${TEST_SOURCES}
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/helpers/termwidth.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_arrayview.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_blockhash.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_byte_source_gzio.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_byte_source_iterators.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_cfdict.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_cvpixelformat.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_env.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_fs.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_fs_attributes.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_fs_execute.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_gif_write.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_glob_stringview.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_halide_io.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_hdf5_io.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_imageformat_options.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_imageview.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_libguid.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_options.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_refcount.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_rocksdb.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_serialization.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_sfinae.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_terminator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_tif_write_multi.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_uri_codec.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_Zinterleaved_io.cpp
    PARENT_SCOPE)

macro(add_imread_test test_name)
    add_test(NAME "${test_name}"
             WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
             COMMAND ./build/imread_tests "*${test_name}*" --durations yes --abortx 10)
endmacro()
