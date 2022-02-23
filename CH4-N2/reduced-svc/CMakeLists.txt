cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(load-model LANGUAGES CXX Fortran)

enable_language(C)
include(FortranCInterface)
FortranCInterface_VERIFY(CXX)

set(CMAKE_CXX_FLAGS "-Ofast")
set(CMAKE_Fortran_FLAGS "-Ofast")

find_package(MPI REQUIRED)
if (MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
else()
    message("No MPI found")
endif()

find_package(GSL REQUIRED)

list(APPEND CMAKE_PREFIX_PATH "/home/artfin/Desktop/neural-networks/project/PES-Fitting-MSA/cpp_pytorch/libtorch/")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include(ExternalProject)

#####################################################################################
ExternalProject_Add(hep
    URL "https://github.com/cschwan/hep-mc/archive/refs/tags/v0.7.tar.gz"
    PREFIX ${CMAKE_SOURCE_DIR}/deps/hep
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory 
                    ${CMAKE_SOURCE_DIR}/deps/hep/src/hep/ 
                    ${CMAKE_SOURCE_DIR}/deps/hep/install/
    UPDATE_DISCONNECTED 1
)  

set(HEP_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/deps/hep/install/include)
#####################################################################################

add_executable(
    svc.exe
    svc.cpp 
    basis_4_2_1_4.f90
    ai_pes_ch4_n2_opt1.hpp
    ai_pes_ch4_n2_opt1.cpp
)

add_dependencies(svc.exe hep)
target_include_directories(svc.exe PRIVATE ${HEP_INCLUDE_DIR})
target_link_libraries(svc.exe "${TORCH_LIBRARIES}" ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
target_link_libraries(svc.exe GSL::gsl GSL::gslcblas)
set_property(TARGET svc.exe PROPERTY CXX_STANDARD 14)
