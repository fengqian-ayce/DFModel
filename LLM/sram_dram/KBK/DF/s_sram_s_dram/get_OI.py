import h5py
import argparse
import matplotlib.pyplot as plt
import numpy as np
import copy

GFLOPS = 307200
DRAM_BW = 100
Net_BW = 50

Memory_Latency = []
Compute_Latency = []
Network_Latency = []
f = h5py.File('log.hdf5', 'r')
Memory_Latency = copy.deepcopy(list(f['Memory_Latency'][:]))
Compute_Latency = copy.deepcopy(list(f['Compute_Latency'][:]))
Network_Latency = copy.deepcopy(list(f['Network_Latency'][:]))
f.close()

num_kernel = len(Memory_Latency)

FLOP = sum(Compute_Latency) * GFLOPS * 0.9
M_Byte = sum(Memory_Latency) * DRAM_BW
N_Byte = sum(Network_Latency) * Net_BW
OI_M = FLOP / M_Byte
OI_N = FLOP / N_Byte


print(OI_M)
print(OI_N)


