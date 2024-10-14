import h5py
import argparse
import matplotlib.pyplot as plt
import numpy as np
import copy


non_attn_names = ['Q', 'K', 'V', 'FFN0', 'FFN1']

attn_flop = 0
non_attn_flop = 0
sram = 0
final_s = 0

f = open('log.txt', 'r')
lines = f.readlines()

for line in lines:
    if line.startswith('SIMD') or line.startswith('SYSTOLIC'):
        if str(line.split()[1]) in non_attn_names:
            non_attn_flop += float(line.split()[-1])
        else:
            attn_flop += float(line.split()[-1])

    if line.startswith('System FLOPS Utilization'):
        util = float(line.split()[-1])

    if line.startswith('final_s'):
        final_s = float(line.split()[-1])

    if line.startswith('SRAM_Per_Config_total[1]'):
        sram = float(line.split()[-1])

f.close()

print(util)
print(final_s)
print(non_attn_flop)
print(attn_flop)

