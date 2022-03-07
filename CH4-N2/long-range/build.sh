#!/bin/sh

set -xe
CXX=g++
CXXFLAGS="-O2"

INCFLAGS="`pkg-config eigen3 --cflags-only-I`"

$CXX $CXXFLAGS $INCFLAGS -c lr_pes_ch4_n2.cpp -o lr_pes_ch4_n2.o
$CXX $CXXFLAGS $INCFLAGS -c main.cpp -o main.o
$CXX $CXXFLAGS main.o lr_pes_ch4_n2.o -o main $LIBS
