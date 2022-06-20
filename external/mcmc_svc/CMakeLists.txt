cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(load-model)

enable_language(C Fortran)

set(CMAKE_CXX_FLAGS "-O2 -ggdb")

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
set(CMAKE_CXX_COMPILER mpicxx)

list(APPEND CMAKE_PREFIX_PATH "/home/artfin/Desktop/neural-networks/project/PES-Fitting-MSA/cpp_pytorch/libtorch/")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(
    svc.exe 
    svc.cc
    mtwist.c
)
target_link_libraries(svc.exe "${TORCH_LIBRARIES}")

target_include_directories(svc.exe PRIVATE "../pes")

add_library(xy4 SHARED IMPORTED)
set_target_properties(xy4 PROPERTIES IMPORTED_LOCATION "/home/artfin/Desktop/neural-networks/project/PES-Fitting-MSA/external/pes/xy4.so") 
target_link_libraries(svc.exe xy4)

add_library(poly SHARED IMPORTED)
set_target_properties(poly PROPERTIES IMPORTED_LOCATION "/home/artfin/Desktop/neural-networks/project/PES-Fitting-MSA/datasets/external/f_basis_4_2_1_4.so")
target_link_libraries(svc.exe poly)
