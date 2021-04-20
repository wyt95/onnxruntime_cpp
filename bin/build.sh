#/bin/bash
export LD_LIBRARY_PATH=../lib
make clean && cmake .. && make

