cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(load-model)

enable_language(C)

set(CMAKE_CXX_FLAGS "-O2 -ggdb")

find_package(MPI REQUIRED)
if (MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
else()
    message("No MPI found")
endif()

list(APPEND CMAKE_PREFIX_PATH "/home/artfin/Desktop/neural-networks/project/PES-Fitting-MSA/cpp_pytorch/libtorch/")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(GSL REQUIRED)

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
    ai_pes_ch4_n2_opt1.hpp
    ai_pes_ch4_n2_opt1.cpp
    svc.cpp
    load_model.hpp
    ../../datasets/external/c_basis_4_2_1_4.cc     
)

add_dependencies(svc.exe hep)
target_include_directories(svc.exe PRIVATE ${HEP_INCLUDE_DIR})
target_link_libraries(svc.exe "${TORCH_LIBRARIES}" ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
target_link_libraries(svc.exe GSL::gsl GSL::gslcblas) 
set_property(TARGET svc.exe PROPERTY CXX_STANDARD 14)
    
