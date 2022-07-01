cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mynn)

enable_language(C)

#set(CMAKE_CXX_FLAGS "-O2 -ggdb")
set(CMAKE_CXX_FLAGS "-O3 -march=native -mfpmath=sse -msse2")
set(CMAKE_C_FLAGS "-O3 -march=native -mfpmath=sse -msse2")

find_package(Eigen3 REQUIRED)
find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})

add_executable(
    nn.exe 
    nn.cpp
    cnpy.h
    cnpy.cpp
    c_basis_4_2_1_4.cc
)

target_link_libraries(nn.exe Eigen3::Eigen ${ZLIB_LIBRARIES})
