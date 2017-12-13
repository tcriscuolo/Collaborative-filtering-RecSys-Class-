CXX=g++
CXXFLAGS=-g -std=c++11 -O2 -pthread
BIN=recomender
SRC=$(wildcard *.cpp)
OBJ=$(SRC:%.cpp=%.o)

all: $(OBJ)
	    $(CXX) -o $(BIN) $^ -pthread
%.o: %.c
	    $(CXX) $@ -c $<

clean:
	    rm -f $(bin) *.o
