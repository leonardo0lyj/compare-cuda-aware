#!/bin/bash
#nvcc -D_DEBUG -g -std=c++11 -Xcompiler -fPIC -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_61,code=sm_61 -I/opt/openmpi-2.0.2-ca/include prog.cpp -c -o prog.o \
#&& /opt/openmpi-2.0.2-ca/bin/mpicxx -g -std=c++11 -L/usr/local/cuda-8.0/lib64 -lcusolver -lcurand -lcublas -lcudart -lcuda -lstdc++ prog.o -o prog.exe \

nvcc -std=c++11 -I/opt/openmpi-2.0.2-ca/include -L/opt/openmpi-2.0.2-ca/lib -lmpi_cxx -lmpi prog.cpp -o prog.exe
