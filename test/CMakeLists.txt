cmake_minimum_required(VERSION 3.0)
project(galois-test)

include_directories(../include)

file(GLOB sources ./*.cc)

foreach(source ${sources})
    get_filename_component(source_we ${source} NAME_WE)
    add_executable(${source_we} ${source})
    target_link_libraries(${source_we} galois "-lz -framework Accelerate")
endforeach(source)
