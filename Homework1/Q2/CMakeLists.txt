cmake_minimum_required(VERSION 3.8)
project(Homework1 LANGUAGES CUDA CXX)

enable_language(CUDA)

find_package(Matplot++ REQUIRED)
find_package(OpenCV)
find_package(Eigen3)

include_directories(${OpenCV_INCLUDE})

# if(CUDA_FOUND)
    add_executable(main main.cpp)
    target_link_libraries(main PUBLIC Matplot++::matplot ${OpenCV_LIBS} Eigen3::Eigen)

    # else()
#     message(STATUS "CUDA NOT FOUND")
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -CUDA_NOT_FOUND")
#     add_executable(main main.cpp helper.cu)
#     target_link_libraries(main ${OpenCV_LIBS})
# endif(CUDA_FOUND)
