import h5py
import argparse
import matplotlib.pyplot as plt
import numpy as np
import copy

GFLOPS = 307200
DRAM_BW = 100
NET_BW = 25
TP = 8

Memory_Latency = []
Network_Latency = []
flop = 0
num_tile = 0
layers = 96
global_batch_size = 2304

simd_or_systolic = []
shard_M = []
shard_K = []
shard_N = []

f = open('log.txt', 'r')
lines = f.readlines()

for line in lines:
    if line.startswith('Memory_Latency['):
        Memory_Latency.append(float(line.split()[-1]))
    if line.startswith('Network_Latency['):
        Network_Latency.append(float(line.split()[-1]))
    if line.startswith('SIMD'):
        simd_or_systolic.append('SIMD')
    if line.startswith('SYSTOLIC'):
        simd_or_systolic.append('SYSTOLIC')
    if line.startswith('shard_M'):
        shard_M.append(float(line.split()[-1]))
    if line.startswith('shard_K'):
        shard_K.append(float(line.split()[-1]))
    if line.startswith('shard_N'):
        shard_N.append(float(line.split()[-1]))
    # if line.startswith('Workload FLOP'):
    #     flop = float(line.split()[-1])
    #     flop = flop / TP / layers / global_batch_size

    if line.startswith('num_tile'):
        num_tile = float(line.split()[-1])

f.close()


for i in range(len(simd_or_systolic)):
    if simd_or_systolic[i] == 'SIMD':
        flop += shard_M[i] * shard_K[i] * shard_N[i] * num_tile
    else:
        flop += 2 * shard_M[i] * shard_K[i] * shard_N[i] * num_tile

mem_bytes = sum(Memory_Latency) * DRAM_BW
net_bytes = sum(Network_Latency) * NET_BW


OI_M = flop / mem_bytes
OI_N = flop / net_bytes


print(OI_M)
print(OI_N)

