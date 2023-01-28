cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)


# We need to explicitly state that we need all CUDA files in the
# particle library to be built with -dc as the member functions
# could be called by other libraries and executables
# set_target_properties( particles PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(main src/main.cu)

#set_property(TARGET particle_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)

#target_link_libraries(particle_test PRIVATE particles)
