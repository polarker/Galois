cmake_minimum_required(VERSION 3.0)

if (UNIX)
  set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O3")
else()
  error("only linux and macos are supported for the moment")
endif()

add_subdirectory(src)
add_subdirectory(example)
add_subdirectory(test)
