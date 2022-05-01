#!/bin/sh

set -xe
CC=gcc

$CC -c -fPIC c_basis_4_2_1_4.cc -o c_basis_4_2_1_4.o
$CC -shared -o c_basis_4_2_1_4.so c_basis_4_2_1_4.o

$CC -c -fPIC c_basis_4_2_1_5.cc -o c_basis_4_2_1_5.o
$CC -shared -o c_basis_4_2_1_5.so c_basis_4_2_1_5.o
