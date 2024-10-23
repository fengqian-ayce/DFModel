import h5py
import argparse
import matplotlib.pyplot as plt
import numpy as np
import copy
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--draft_model', type=str, required=True)
parser.add_argument('--target_model', type=str, required=True)
parser.add_argument('--k', type=int, required=True)
parser.add_argument('--acceptance_rate', type=float, required=True)

args = parser.parse_args()

draft_model = args.draft_model
target_model = args.target_model
k = args.k
acceptance_rate = args.acceptance_rate




f = open(draft_model, 'r')
lines = f.readlines()

for line in lines:
    if line.startswith('final_ii_s'):
        ii_draft = float(line.split()[-1])
    
    if line.startswith('TP '):
        TP_draft = float(line.split()[-1])

    if line.startswith('PP '):
        PP_draft = float(line.split()[-1])

    if line.startswith('num_micro_batch_per_pipeline '):
        bs_draft = float(line.split()[-1])
    
f.close()



f = open(target_model, 'r')
lines = f.readlines()

for line in lines:
    if line.startswith('final_ii_s'):
        ii_target = float(line.split()[-1])
    
    if line.startswith('TP '):
        TP_target = float(line.split()[-1])

    if line.startswith('PP '):
        PP_target = float(line.split()[-1])

    if line.startswith('num_micro_batch_per_pipeline '):
        bs_target = float(line.split()[-1])
    
f.close()

ii_draft = ii_draft / bs_draft
ii_target = ii_target / bs_target

if TP_draft != TP_target or PP_draft != PP_target:
    raise Exception()


tokens_per_s = (k * acceptance_rate + 1) / (ii_draft * k + ii_target)

print('tokens_per_s', tokens_per_s)
print('ii_draft', ii_draft)
print('ii_target', ii_target)

