cmake_minimum_required(VERSION 2.8)
project(lexical_cast)

install(DIRECTORY include/boost/ DESTINATION include/boost
  FILES_MATCHING PATTERN "*.hh")
install(DIRECTORY include/boost/ DESTINATION include/boost
  FILES_MATCHING PATTERN "*.hpp")
