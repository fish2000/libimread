# Author: Alexander Böhn
# © 2011.10 -- GPL, Motherfuckers

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

set(TEST_SOURCES
  ${TEST_SOURCES}
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/test_apple_io.mm
  ${CMAKE_CURRENT_LIST_DIR}/test_fs.cpp
  ${CMAKE_CURRENT_LIST_DIR}/test_gif_write.cpp
  ${CMAKE_CURRENT_LIST_DIR}/test_halide_io.cpp
  ${CMAKE_CURRENT_LIST_DIR}/test_interleaved_io.cpp
  ${CMAKE_CURRENT_LIST_DIR}/test_objc_rt.mm
  ${CMAKE_CURRENT_LIST_DIR}/test_sfinae.cpp
  ${CMAKE_CURRENT_LIST_DIR}/test_terminator.cpp
PARENT_SCOPE)