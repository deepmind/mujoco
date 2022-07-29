# This Makefile assumes that you have GLFW libraries and headers installed on,
# which is commonly available through your distro's package manager.
# On Debian and Ubuntu, GLFW can be installed via `apt install libglfw3-dev`.

COMMON=-O2 -I../include -L../lib -std=c++17 -pthread -Wl,-no-as-needed -Wl,-rpath,'$$ORIGIN'/../lib

all:
	$(CXX) $(COMMON) testxml.cc            -lmujoco          -o ../bin/testxml
	$(CXX) $(COMMON) testspeed.cc          -lmujoco          -o ../bin/testspeed
	$(CXX) $(COMMON) compile.cc            -lmujoco          -o ../bin/compile
	$(CXX) $(COMMON) derivative.cc         -lmujoco -fopenmp -o ../bin/derivative
	$(CXX) $(COMMON) basic.cc              -lmujoco -lglfw   -o ../bin/basic
	$(CXX) $(COMMON) record.cc             -lmujoco -lglfw   -o ../bin/record
