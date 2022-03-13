cmake_minimum_required(VERSION 3.0)
project(edge_extraction)

set(CMAKE_BUILD_TYPE "Release")                         # 编译模式 Debug/Release
set(CMAKE_CXX_FLAGS "-std=c++11")                       # 针对C++的编译选项
set(CMAKE_CXX_FLAGS_DEBUG "-O1 -Wall -g -pthread")      # 针对C++在Debug模式下的编译选项
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")    # 针对C++在Release模式下的编译选项
# 采用如下方式也可，但是编译选项针对所有类型编译器
# add_compile_options(-std=c++11 -O3 -Wall -g -pthread) # 不推荐

find_package(Eigen3 REQUIRED QUIET)
find_package(PCL REQUIRED QUIET)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

add_executable(extract_edge src/main.cpp)
target_link_libraries(extract_edge
    ${PCL_LIBRARIES}
)

add_executable(visualize_edge src/main_visualization.cpp)
target_link_libraries(visualize_edge
    ${PCL_LIBRARIES}
)
