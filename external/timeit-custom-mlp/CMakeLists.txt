cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# see https://stackoverflow.com/questions/45933732/how-to-specify-a-compiler-in-cmake
# compilers should be set BEFORE a call to `project` which looks for a default compiler executable
set(CMAKE_C_COMPILER "/share/apps/tools/gcc/9.2.0/bin/gcc")
set(CMAKE_CXX_COMPILER "/share/apps/tools/gcc/9.2.0/bin/g++")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(mynn LANGUAGES C CXX Fortran)

set(CMAKE_Fortran_FLAGS "-Ofast -march=native")
set(CMAKE_CXX_FLAGS "-Ofast -march=native -mfpmath=sse -msse2")
set(CMAKE_C_FLAGS "-Ofast -march=native -mfpmath=sse -msse2")
#set(CMAKE_C_FLAGS "-O2 -ggdb")
#set(CMAKE_CXX_FLAGS "-O2 -ggdb")

find_package(OpenMP REQUIRED COMPONENTS C CXX)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  #message("OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
  #message("CMake_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
endif()

find_package(MPI REQUIRED COMPONENTS C CXX)
if (MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
else()
    message("No MPI found")
endif()

find_package(Eigen3 REQUIRED CONFIG)
find_package(ZLIB REQUIRED)
#find_package(GSL REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})

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
    c_basis_4_2_1_4_intermolecular.cc
    c_basis_4_2_1_4_intermolecular.h
    c_jac_4_2_1_4_intermolecular.cc
    c_jac_4_2_1_4_intermolecular.h
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
    c_basis_1_1_1_2_3_1_3.h
    c_jac_1_1_1_2_3_1_3.cc
    c_jac_1_1_1_2_3_1_3.h
)

add_executable(
    n2-ar-nonrigid.exe
    n2-ar-nonrigid.cpp
    cnpy.h
    cnpy.cpp
    mtwist.h
    mtwist.c
    mlp.hpp
    scaler.hpp
    scaler.cpp
    c_basis_2_1_4_purify.cc
    c_basis_2_1_4_purify.h
    c_jac_2_1_4_purify.cc
    c_jac_2_1_4_purify.h
)

add_executable(
    n2-ar-svc.exe
    n2-ar-svc.cpp
    cnpy.h
    cnpy.cpp
    mlp.hpp
    scaler.hpp
    scaler.cpp
    c_basis_2_1_4_purify.cc
    c_basis_2_1_4_purify.h
    c_jac_2_1_4_purify.cc
    c_jac_2_1_4_purify.h
)

add_executable(
    ch4-n2-svc.exe
    ch4-n2-svc.cpp
    ../pes/xy4.f90
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

target_link_libraries(ch4-n2-nonrigid.exe PRIVATE Eigen3::Eigen ${ZLIB_LIBRARIES})
target_link_libraries(ch4-n2-rigid.exe    PRIVATE Eigen3::Eigen ${ZLIB_LIBRARIES})
target_link_libraries(ethanol.exe         PRIVATE Eigen3::Eigen ${ZLIB_LIBRARIES} ${OpenMP_CXX_FLAGS})
target_link_libraries(n2-ar-nonrigid.exe  PRIVATE Eigen3::Eigen ${ZLIB_LIBRARIES})
target_link_libraries(n2-ar-svc.exe       PRIVATE Eigen3::Eigen ${ZLIB_LIBRARIES} ${MPI_CXX_LIBRARIES})
target_link_libraries(ch4-n2-svc.exe      PRIVATE Eigen3::Eigen ${ZLIB_LIBRARIES} ${MPI_CXX_LIBRARIES} -static-libstdc++)
