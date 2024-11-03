set(CMAKE_CUDA_STANDARD 11)
enable_language(CUDA)

find_package(CUDAToolkit)
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(CudaTest VectorAdd.cu)

add_executable(Hello hello.cpp)



