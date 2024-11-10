cmake_minimum_required (VERSION 3.8)

project (CudaTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)  
set(CMAKE_CXX_STANDARD_REQUIRED ON)  
set(CMAKE_CUDA_ARCHITECTURES "70")

add_subdirectory(Test/)
add_subdirectory(Source/LinearAlgebra/)