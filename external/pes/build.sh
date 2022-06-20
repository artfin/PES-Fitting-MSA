#!/bin/sh

set -xe

F=gfortran
FFLAGS="-fPIC -shared -fdefault-real-8"

$F $FFLAGS xy4.f90 -o xy4.so

CC=gcc
CCFLAGS="-O2"
FFLAGS="-O2"
LDFLAGS="-Wl,--copy-dt-needed-entries -lgfortran -lstdc++"

$CC $CCFLAGS -c bind_ch4.cc -o bind_ch4.o
$F $FFLAGS -c xy4.f90 -o xy4.o 
$CC bind_ch4.o xy4.o -o bind_ch4.exe $LDFLAGS
