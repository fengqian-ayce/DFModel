import h5py
import argparse
import matplotlib.pyplot as plt
import numpy as np
import copy


non_attn_names = ['Proj1', 'Proj2', 'Proj3', 'Conv', 'Multiply']

attn_flop = 0
non_attn_flop = 0
attn_latency = 0
non_attn_latency = 0
sram = 0

f = open('log.txt', 'r')
lines = f.readlines()

for line in lines:
    if line.startswith('SIMD') or line.startswith('SYSTOLIC'):
        if str(line.split()[1]) in non_attn_names:
            non_attn_flop += float(line.split()[-1])
        else:
            attn_flop += float(line.split()[-1])

    if line.startswith('Per_Config_II['):
        if line.startswith('Per_Config_II[1]'):
            attn_latency += float(line.split()[-1])
        else:
            non_attn_latency += float(line.split()[-1])

    if line.startswith('System FLOPS Utilization'):
        util = float(line.split()[-1])

    if line.startswith('final_s'):
        final_s = float(line.split()[-1])

    if line.startswith('SRAM_Per_Config_total[1]'):
        sram += float(line.split()[-1])

f.close()

print(util)
print(final_s)
print(non_attn_flop)
print(attn_flop)
print(non_attn_latency)
print(attn_latency)
print(sram/1024**2)

