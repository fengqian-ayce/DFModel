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
Per_Config_II = []

f = open("log.txt", 'r')
lines = f.readlines()

for line in lines:
    if line.startswith('Network_Latency['):
        Network_Latency.append(float(line.split()[-1]))

    if line.startswith('Memory_Latency['):
        Memory_Latency.append(float(line.split()[-1]))

    if line.startswith('Compute_Latency['):
        Compute_Latency.append(float(line.split()[-1]))
    
    if line.startswith('Workload FLOP'):
        FLOP = float(line.split()[-1])
    
    if line.startswith('Per-Accelerator Throughput (GFLOPS)'):
        GFLOPS = float(line.split()[-1])

    if line.startswith('TP '):
        TP = float(line.split()[-1])

    if line.startswith('PP '):
        PP = float(line.split()[-1])
    
    if line.startswith('DP '):
        DP = float(line.split()[-1])

f.close()






total_latency = 0
for i in range(len(Compute_Latency)):
    total_latency += max(Compute_Latency[i], Memory_Latency[i], Network_Latency[i])


for a, b, c in zip(Compute_Latency, Memory_Latency, Network_Latency):
    print(a, b, c)

print('util', FLOP / total_latency / (TP * PP * DP * GFLOPS))


