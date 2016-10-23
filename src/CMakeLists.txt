cmake_minimum_required(VERSION 3.0)
project(galois)

include_directories(../include)
file(GLOB_RECURSE sources ./*.cc)

add_library(galois ${sources})
