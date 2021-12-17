COMMON=-O2 -I../include -L../bin -std=c++11 -mavx -pthread -Wl,-rpath,'$$ORIGIN'

.PHONY: all
all:
	g++ $(COMMON) testxml.cc    -lmujoco210nogl                             -o ../bin/testxml
	g++ $(COMMON) testspeed.cc  -lmujoco210nogl                             -o ../bin/testspeed
	g++ $(COMMON) compile.cc    -lmujoco210nogl                             -o ../bin/compile
	g++ $(COMMON) derivative.cc -lmujoco210nogl -fopenmp                    -o ../bin/derivative
	g++ $(COMMON) basic.cc      -lmujoco210 -lGL -lglew ../bin/libglfw.so.3 -o ../bin/basic
	g++ $(COMMON) record.cc     -lmujoco210 -lGL -lglew ../bin/libglfw.so.3 -o ../bin/record
	gcc -c -O2 -mavx -I../include ../include/uitools.c
	g++ $(COMMON) uitools.o simulate.cc -lmujoco210 -lGL -lglew ../bin/libglfw.so.3 -o ../bin/simulate
	rm *.o
