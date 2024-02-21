cmake_minimum_required(VERSION 3.0)
project(MatrixMultiplication)

# Use find_package to locate Eigen
find_package(Eigen3 REQUIRED)

# Include Eigen directories
include_directories(${EIGEN3_INCLUDE_DIR})

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add CUDA support
find_package(CUDA REQUIRED)
# Specify CUDA architecture (optional)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_75,code=sm_75)  # Adjust as needed

# Define the executable and its source files
cuda_add_executable(matmul matmul.cpp matmul.cu)

# Link against CUDA libraries
target_link_libraries(matmul ${CUDA_LIBRARIES})

# Define the executable and its source files
cuda_add_executable(max max.cu)

# Link against CUDA libraries
target_link_libraries(max ${CUDA_LIBRARIES})
