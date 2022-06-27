cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(timeit)

enable_language(C)

#set(CMAKE_CXX_FLAGS "-O2 -ggdb")
set(CMAKE_CXX_FLAGS "-O3 -march=native -mfpmath=sse -msse2")
set(CMAKE_C_FLAGS "-O3 -march=native -mfpmath=sse -msse2")

list(APPEND CMAKE_PREFIX_PATH "/home/artfin/Desktop/neural-networks/project/PES-Fitting-MSA/cpp_pytorch/libtorch/")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(GSL REQUIRED)

add_executable(
    timeit.exe 
    ai_pes_ch4_n2_opt1.hpp
    ai_pes_ch4_n2_opt1.cpp
    timeit.cpp
    load_model.hpp
    ../../datasets/external/c_basis_4_2_1_4.cc     
)

target_link_libraries(timeit.exe GSL::gsl GSL::gslcblas) 
target_link_libraries(timeit.exe "${TORCH_LIBRARIES}")
set_property(TARGET timeit.exe PROPERTY CXX_STANDARD 14)
