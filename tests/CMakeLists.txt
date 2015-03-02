
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

set(TEST_SOURCES
  ${TEST_SOURCES}
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/test_halide_shared_buffer.cpp
PARENT_SCOPE)