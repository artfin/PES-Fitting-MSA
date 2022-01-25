.PHONY: all clean

all: main

main: main.o basis.o
	g++ -o main main.o basis.o

main.o: main.cpp
	g++ -ggdb -O2 -c main.cpp -I/usr/local/include/eigen3/ 

basis-maxord-4.o: basis-maxord-4.f90
	gfortran -fdefault-real-8 -O1 -c basis-maxord-4.f90 

basis-maxord-4.so: basis-maxord-4.f90
	gfortran -shared basis-maxord-4.f90 -o basis-maxord-4.so

basis-maxord-2.so: basis-maxord-2.f90
	gfortran -shared basis-maxord-2.f90 -o basis-maxord-2.so

clean:
	rm -vf basis-maxord-4.so
	rm -vf basis-maxord-2.so
	rm -vf basis-maxord-4.o
	rm -vf basis.mod
	rm -vf main.o
	rm -vf main
