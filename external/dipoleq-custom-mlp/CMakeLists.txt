cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# see https://stackoverflow.com/questions/45933732/how-to-specify-a-compiler-in-cmake
# compilers should be set BEFORE a call to `project` which looks for a default compiler executable
#set(CMAKE_C_COMPILER "/share/apps/tools/gcc/9.2.0/bin/gcc")
#set(CMAKE_CXX_COMPILER "/share/apps/tools/gcc/9.2.0/bin/g++")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(mynn LANGUAGES C CXX Fortran)

#set(CMAKE_Fortran_FLAGS "-Ofast -march=native")
#set(CMAKE_CXX_FLAGS "-Ofast -march=native -mfpmath=sse -msse2")
#set(CMAKE_C_FLAGS "-Ofast -march=native -mfpmath=sse -msse2")
set(CMAKE_C_FLAGS "-O2 -ggdb")
set(CMAKE_CXX_FLAGS "-O2 -ggdb")

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
find_package(GSL REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})

add_executable(
    2b-n2-ar-qdipole.exe
    2b-n2-ar-qdipole.cpp
    cnpy.h
    cnpy.cpp
    qmodel.hpp
    scaler.hpp
    scaler.cpp
    c_basis_2_1_4_purify.cc
    c_basis_2_1_4_purify.h
    c_basis_1_1_1_4_purify.cc
    c_basis_1_1_1_4_purify.h
)

target_link_libraries(2b-n2-ar-qdipole.exe PRIVATE Eigen3::Eigen ${ZLIB_LIBRARIES})
