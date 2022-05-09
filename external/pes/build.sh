#!/bin/sh

set -xe

F=gfortran
FFLAGS="-fPIC -shared -fdefault-real-8"

$F $FFLAGS xy4.f90 -o xy4.so
