cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O3")

add_subdirectory(src)
add_subdirectory(example)
add_subdirectory(test)
