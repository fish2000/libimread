cmake_minimum_required(VERSION 2.8)
project(lrucache11)

# set(CMAKE_CXX_STANDARD 14)
# add_definitions(-std=c++14)

install(DIRECTORY lrucache11 DESTINATION include
  FILES_MATCHING PATTERN "*.hh")
install(DIRECTORY lrucache11 DESTINATION include
  FILES_MATCHING PATTERN "*.hpp")
