cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(radial)

enable_language(C Fortran)

set(CMAKE_CXX_FLAGS "-O2 -ggdb")

list(APPEND CMAKE_PREFIX_PATH "/home/artfin/Desktop/neural-networks/project/PES-Fitting-MSA/cpp_pytorch/libtorch/")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(
    radial.exe 
    main.cpp
    ai_pes_ch4_n2_opt1.cpp 
)

target_link_libraries(radial.exe "${TORCH_LIBRARIES}")

add_library(poly SHARED IMPORTED)
set_target_properties(poly PROPERTIES IMPORTED_LOCATION "/home/artfin/Desktop/neural-networks/project/PES-Fitting-MSA/datasets/external/f_basis_4_2_1_4.so")
target_link_libraries(radial.exe poly)

find_package(GSL REQUIRED)
target_link_libraries(radial.exe GSL::gsl GSL::gslcblas) 
