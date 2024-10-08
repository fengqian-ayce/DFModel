import h5py
import argparse
import matplotlib.pyplot as plt
import numpy as np
import copy


attn_names = ['MHA_GEMM_1', 'SOFTMAX', 'MHA_GEMM_2']
non_attn_names = ['Q', 'K', 'V', 'FFN0', 'FFN1']

attn = 0
non_attn = 0

f = open('log.txt', 'r')
lines = f.readlines()

for line in lines:
    if line.startswith('SIMD') or line.startswith('SYSTOLIC'):
        if str(line.split()[1]) in attn_names:
            attn += float(line.split()[-1])
        elif str(line.split()[1]) in non_attn_names:
            non_attn += float(line.split()[-1])

    if line.startswith('System FLOPS Utilization'):
        util = float(line.split()[-1])
    if line.startswith('final_s'):
        latency = float(line.split()[-1])

f.close()

print(attn)
print(non_attn)
print(attn+non_attn)
print(util)
print(latency)
