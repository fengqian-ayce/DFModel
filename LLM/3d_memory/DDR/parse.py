import h5py
import argparse
import matplotlib.pyplot as plt
import numpy as np
import copy
import sys


aaa = ['0.2', '0.35', '0.5', '0.65', '0.8']

for a in aaa:
    # print('--------------', a, '-----------')
    f = open(a+'/log.txt', 'r')
    lines = f.readlines()

    for line in lines:
        if line.startswith('System FLOPS Utilization'):
            util = float(line.split()[-1])
            print(util)
            
    f.close()