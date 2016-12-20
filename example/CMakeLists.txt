cmake_minimum_required(VERSION 3.0)
project(galois-example)

include_directories(../include)

file(GLOB sources ./*.cc)

foreach(source ${sources})
    get_filename_component(source_we ${source} NAME_WE)
    add_executable(${source_we} ${source})
    if (APPLE)
        target_link_libraries(${source_we} galois "-lz -framework Accelerate")
    elseif (UNIX)
        target_link_libraries(${source_we} galois "-lz -lopenblas")
    else()
        error("only linux and macos are supported for the moment")
    endif()
endforeach(source)
