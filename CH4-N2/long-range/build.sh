#!/bin/sh

set -xe
CC=gcc
CXX=g++
CXXFLAGS="-O2"

INCFLAGS="`pkg-config eigen3 --cflags-only-I`"

$CXX $CXXFLAGS $INCFLAGS -c export.cpp -o export.o
$CXX $CXXFLAGS $INCFLAGS -c lr_pes_ch4_n2.cpp -o lr_pes_ch4_n2.o
$CXX $CXXFLAGS export.o lr_pes_ch4_n2.o -o export.exe $LIBS


