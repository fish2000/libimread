# Author: Alexander Böhn
# © 2011.10 -- GPL, Motherfuckers

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

set(TEST_SOURCES
  ${TEST_SOURCES}
  ${CMAKE_CURRENT_LIST_DIR}/helpers/IMTestReceiver.m
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/test_apple_io.mm
  ${CMAKE_CURRENT_LIST_DIR}/test_blockhash.cpp
  ${CMAKE_CURRENT_LIST_DIR}/test_fs.cpp
  ${CMAKE_CURRENT_LIST_DIR}/test_gif_write.cpp
  ${CMAKE_CURRENT_LIST_DIR}/test_halide_io.cpp
  ${CMAKE_CURRENT_LIST_DIR}/test_hdf5_io.cpp
  # ${CMAKE_CURRENT_LIST_DIR}/test_json_block_traverse.mm
  # ${CMAKE_CURRENT_LIST_DIR}/test_libguid.mm
  ${CMAKE_CURRENT_LIST_DIR}/test_objc_rt.mm
  # ${CMAKE_CURRENT_LIST_DIR}/test_refcount.mm
  # ${CMAKE_CURRENT_LIST_DIR}/test_sfinae.mm
  ${CMAKE_CURRENT_LIST_DIR}/test_sszip.mm
  ${CMAKE_CURRENT_LIST_DIR}/test_terminator.mm
  ${CMAKE_CURRENT_LIST_DIR}/test_Zinterleaved_io.cpp
PARENT_SCOPE)