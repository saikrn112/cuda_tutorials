############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################
cmake_minimum_required(VERSION 3.16)

# collect source files
project(cmake_and_cuda LANGUAGES CXX CUDA)
# check_language(CUDA)
find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW2_files main.cpp reference_calc.cpp compare.cpp)
    
ADD_EXECUTABLE(HW2 ${HW2_files} ${hdr} ${cu})


# Link against OpenCV libraries
target_link_libraries(HW2 ${OpenCV_LIBS})