cmake_minimum_required(VERSION 3.16)
project(aco_tsp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenMP REQUIRED)

add_executable(aco
  src/main.cpp
)

target_link_libraries(aco PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(aco PRIVATE -O3 -march=native -Wall -Wextra)
