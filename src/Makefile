.PHONY: clean depend

CC := g++
CFLAG := -O2 -Wall

LOCAL_LIB_PATH = ./lib
LLIB = -L${LOCAL_LIB_PATH} -lcppmol -lcppmat
GLIB = -lm -llapack

all: msa 

msa: msa.o monomial.o polynomial.o
	${CC} ${CFLAG} -o $@ $^

%: %.o
	${CC} ${CFLAG} -o $@ $^ ${INCLUDE} ${LLIB} ${GLIB}
-include .depend

%.o: %.cpp
	${CC} ${CFLAG} -c $< ${INCLUDE}

clean:
	rm -rf *~  *.o  msa 

depend: 
	${CC} -MM *.cpp ${INCLUDE} > .depend
