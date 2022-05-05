#!/bin/sh

set -xe

F=gfortran
FFLAGS="-fPIC -shared -fdefault-real-8"

CC=gcc
CCFLAGS="-Ofast -ggdb"
LDFLAGS="-Wl,--copy-dt-needed-entries -lgfortran -lstdc++"

mkdir -p obj/
$F $FFLAGS xy4.f90 -o obj/xy4.so
$CC $CCFLAGS -c sampler.cc -o obj/sampler.o
$CC $CCFLAGS -c mtwist.c -o obj/mtwist.o 
$CC obj/sampler.o obj/xy4.so obj/mtwist.o -o program.exe $LDFLAGS
