#!/bin/bash
CUDA_VISIBLE_DEVICES=0 /opt/openmpi-2.0.2-ca/bin/mpirun --host 192.17.100.214,192.17.100.215,192.17.100.216 \
 -np 3 -npernode 1 ./prog.exe
