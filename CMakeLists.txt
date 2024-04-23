cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
Project(CUDA_Freshman CXX C CUDA)

set(CMAKE_CUDA_FLAGS "-arch=compute_75 -g -G -O3")

include_directories(./include)
add_subdirectory(hello_world)
add_subdirectory(check_dimension)
add_subdirectory(device_information)
add_subdirectory(grid_block)
add_subdirectory(sum_arrays)
add_subdirectory(sum_arrays_timer)
add_subdirectory(pthread_index)
add_subdirectory(test)




