#/bin/bash

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/artfin/Desktop/neural-networks/project/onnxruntime/build/Linux/RelWithDebInfo/
#sudo ldconfig # - to update LD_LIBRARY_PATH in the linker runtime

set -xe

EXE=infer.exe

CXXFLAGS="-O2 -march=native -mtune=native"
CFLAGS="-O2 -march=native -mtune=native"

INC_ONNX="-I/home/artfin/Desktop/neural-networks/project/onnxruntime/include/onnxruntime/core/session/"
LINK_ONNX="-L/home/artfin/Desktop/neural-networks/project/onnxruntime/build/Linux/RelWithDebInfo/ -lonnxruntime"

g++ $CXXFLAGS $INC_ONNX -c inference.cpp -o inference.o
gcc $CFLAGS -c c_basis_4_2_1_4.cc -o c_basis_4_2_1_4.o

g++ inference.o c_basis_4_2_1_4.o -o $EXE $LINK_ONNX
