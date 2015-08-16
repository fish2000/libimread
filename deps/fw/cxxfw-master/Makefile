CXXFLAGS += -std=c++11

test: hash.o test.o
	${CXX} -o $@ $^

test.o: array.h dictionary.h hash.h
