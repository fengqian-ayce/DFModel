import h5py
import argparse
import matplotlib.pyplot as plt
import numpy as np
import copy




f = open('log.txt', 'r')
lines = f.readlines()

for line in lines:
    if line.startswith('final_ii_s'):
        final_ii_s = float(line.split()[-1])
    
    if line.startswith('tile_size '):
        tile_size = float(line.split()[-1])

    if line.startswith('num_tile '):
        num_tile = float(line.split()[-1])

    if line.startswith('layer_per_stage '):
        layer_per_stage = float(line.split()[-1])
    
    if line.startswith('TP '):
        TP = float(line.split()[-1])

    if line.startswith('PP '):
        PP = float(line.split()[-1])

    if line.startswith('Compute_Latency['):
        Compute_Latency = float(line.split()[-1])

    if line.startswith('Memory_Latency['):
        Memory_Latency = float(line.split()[-1])
    
    if line.startswith('serialization_latency_allreduce_node['):
        serialization_latency_allreduce_node = float(line.split()[-1])
    
    if line.startswith('link_latency_allreduce_node['):
        link_latency_allreduce_node = float(line.split()[-1])

f.close()






hidden = 4096
seq_len = 4096
link_latency = 150
link_bw = 50
word = 2

Compute_Latency = Compute_Latency * layer_per_stage / 1e6
Memory_Latency = Memory_Latency * layer_per_stage / 1e6
serialization_latency_allreduce_node = serialization_latency_allreduce_node * layer_per_stage / 1e6
link_latency_allreduce_node = link_latency_allreduce_node * layer_per_stage / 1e6
II = Compute_Latency + Memory_Latency + serialization_latency_allreduce_node + link_latency_allreduce_node



print(1 / (II / 1e3) * num_tile * tile_size) # tokens/s
print(II) # II ms
print(Compute_Latency)
print(Memory_Latency)
print(serialization_latency_allreduce_node)
print(link_latency_allreduce_node)

print()

print(Compute_Latency * PP \
      + Memory_Latency * PP \
      + serialization_latency_allreduce_node * PP \
      + link_latency_allreduce_node * PP \
      + (PP-1) * hidden*seq_len/TP*word /link_bw/1e6 \
      + (PP-1) * link_latency/1e6) # latency (ms)
print(Compute_Latency * PP)
print(Memory_Latency * PP)
print(serialization_latency_allreduce_node * PP)
print(link_latency_allreduce_node * PP)
print((PP-1) * hidden*seq_len/TP*word /link_bw/1e6)
print((PP-1) * link_latency/1e6)

