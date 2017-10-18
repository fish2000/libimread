cmake_minimum_required(VERSION 2.8)
project(lmdbxx)

find_package(LMDB REQUIRED)

# set(CMAKE_CXX_STANDARD 17)
# add_definitions(-std=c++17)

install(DIRECTORY lmdbxx DESTINATION include
  FILES_MATCHING PATTERN "*.hh")
install(DIRECTORY lmdbxx DESTINATION include
  FILES_MATCHING PATTERN "*.h")
