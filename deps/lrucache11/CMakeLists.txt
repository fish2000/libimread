cmake_minimum_required(VERSION 2.8)
project(lrucache11)

# set(CMAKE_CXX_STANDARD 17)
# add_definitions(-std=c++17)

install(DIRECTORY lrucache11 DESTINATION include
  FILES_MATCHING PATTERN "*.hh")
install(DIRECTORY lrucache11 DESTINATION include
  FILES_MATCHING PATTERN "*.hpp")
