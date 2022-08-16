cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mynn LANGUAGES Fortran C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED CONFIG)
find_package(ZLIB REQUIRED)
find_package(GSL REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})

#set(CMAKE_CXX_FLAGS "-O2 -ggdb")
set(CMAKE_CXX_FLAGS "-O3 -march=native -mfpmath=sse -msse2")
set(CMAKE_C_FLAGS "-O3 -march=native -mfpmath=sse -msse2")

add_executable(
    ch4-n2-nonrigid.exe 
    ch4-n2-nonrigid.cpp
    cnpy.h
    cnpy.cpp
    mlp.hpp
    scaler.hpp
    scaler.cpp
    c_basis_4_2_1_4_purify.h
    c_basis_4_2_1_4_purify.cc
    c_jac_4_2_1_4_purify.h
    c_jac_4_2_1_4_purify.cc
)

add_executable(
    ch4-n2-rigid.exe
    ch4-n2-rigid.cpp
    cnpy.h
    cnpy.cpp
    mlp.hpp
    scaler.hpp
    scaler.cpp
    c_basis_4_2_1_4_intermolecular.h
    c_basis_4_2_1_4_intermolecular.cc
    c_jac_4_2_1_4_intermolecular.h
    c_jac_4_2_1_4_intermolecular.cc
    ai_pes_ch4_n2_opt1.cpp
    ai_pes_ch4_n2_opt1.hpp
)

add_executable(
    ethanol.exe
    ethanol.cpp
    cnpy.h
    cnpy.cpp
    mlp.hpp
    scaler.hpp
    scaler.cpp
    c_basis_1_1_1_2_3_1_3.cc
    c_jac_1_1_1_2_3_1_3.cc
)

target_link_libraries(ch4-n2-nonrigid.exe  PRIVATE Eigen3::Eigen ${ZLIB_LIBRARIES} GSL::gsl GSL::gslcblas)
target_link_libraries(ch4-n2-rigid.exe     PRIVATE Eigen3::Eigen ${ZLIB_LIBRARIES} GSL::gsl GSL::gslcblas)
target_link_libraries(ethanol.exe          PRIVATE Eigen3::Eigen ${ZLIB_LIBRARIES} GSL::gsl GSL::gslcblas)
