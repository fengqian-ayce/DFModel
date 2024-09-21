import h5py
import argparse
import matplotlib.pyplot as plt
import numpy as np
import copy

f = open('log.txt', 'r')
lines = f.readlines()
for line in lines:
    if line.startswith('GFLOPS'):
        GFLOPS = float(line.split()[-1])
    if line.startswith('DRAM_BW'):
        DRAM_BW = float(line.split()[-1])
f.close()

Memory_Latency = []
Compute_Latency = []
f = h5py.File('log.hdf5', 'r')
Memory_Latency = copy.deepcopy(list(f['Memory_Latency'][:]))
Compute_Latency = copy.deepcopy(list(f['Compute_Latency'][:]))
f.close()
num_kernel = len(Memory_Latency)

FLOP = sum(Compute_Latency) * GFLOPS * 0.9
Byte = sum(Memory_Latency) * DRAM_BW 
OI = FLOP / Byte

print(FLOP)
print(Byte)
print(OI)
