import h5py
import argparse
import matplotlib.pyplot as plt
import numpy as np
import copy



accelerators = ['h100', 'tpu', 'sn30', 'wse-2']
topologies = ['2d_torus', 'dragonfly', '3d_torus', 'dgx_1', 'dgx_2']
combinations = ['ddr_nvlink', 'hbm_pcie', 'hbm_nvlink', 'ddr_pcie']



mem_dict = {}
com_dict = {}
net_dict = {}
bottle_dict = {}

for combination in combinations:
    for accelerator in accelerators:
        for topology in topologies:
            name = accelerator+'/'+topology+'_'+combination
            
            Memory_Latency = []
            Compute_Latency = []
            Network_Latency = []
            bottlenecks = []

            
            f = h5py.File(name+'/log.hdf5', 'r')
            Memory_Latency = copy.deepcopy(list(f['Memory_Latency'][:]))
            Compute_Latency = copy.deepcopy(list(f['Compute_Latency'][:]))
            Network_Latency = copy.deepcopy(list(f['Network_Latency'][:]))
            f.close()
            
            num_kernel = len(Memory_Latency)
            
            
            for i in range(num_kernel):
                maxV = max(Memory_Latency[i], Compute_Latency[i], Network_Latency[i])
                if maxV == Network_Latency[i]:
                    bottlenecks.append('n')
                elif maxV == Compute_Latency[i]:
                    bottlenecks.append('c') 
                else:
                    bottlenecks.append('m')
                    
            
            mem_dict[name] = Memory_Latency
            com_dict[name] = Compute_Latency
            net_dict[name] = Network_Latency
            bottle_dict[name] = bottlenecks






total_value = -1
for m in range(len(combinations)):
    for i in range(len(accelerators)):
        for j in range(len(topologies)):
            name = accelerators[i]+'/'+topologies[j]+'_'+combinations[m]
            
            Memory_Latency = mem_dict[name]
            Compute_Latency = com_dict[name]
            Network_Latency = net_dict[name]
            bottlenecks = bottle_dict[name]
            num_kernel = len(Memory_Latency)
            
            cumulative_value = [0]
            value = []
            color = []
            for k in range(num_kernel):
                if bottlenecks[k] == 'm':
                    value.append(Memory_Latency[k])
                    color.append('r')
                elif bottlenecks[k] == 'c':
                    value.append(Compute_Latency[k])
                    color.append('g')
                elif bottlenecks[k] == 'n':
                    value.append(Network_Latency[k])
                    color.append('b')
            
            for k in range(1, num_kernel):
                cumulative_value.append(value[k-1] + cumulative_value[k-1])
            
            tmp_total_value = sum(value)
            if tmp_total_value > total_value:
                total_value = tmp_total_value
                setting_w_greatest_value = name

print(total_value)
print(setting_w_greatest_value)

gadget = ['Latency Breakdown (DDR & PCIe)', 'Latency Breakdown (DDR & NVLink)', 'Latency Breakdown (HBM & PCIe)', 'Latency Breakdown (HBM, & NVLink)']



for m in range(len(combinations)):
    fig, ax = plt.subplots(4, 5)
    fig.set_figheight(12)
    fig.set_figwidth(15)
    fig.suptitle(gadget[m], fontsize=40)
    
    for i in range(len(accelerators)):
        for j in range(len(topologies)):
            name = accelerators[i]+'/'+topologies[j]+'_'+combinations[m]
            
            Memory_Latency = mem_dict[name]
            Compute_Latency = com_dict[name]
            Network_Latency = net_dict[name]
            bottlenecks = bottle_dict[name]
            num_kernel = len(Memory_Latency)
            
            cumulative_value = [0]
            value = []
            color = []
            for k in range(num_kernel):                    
                if bottlenecks[k] == 'm':
                    value.append(Memory_Latency[k])
                    color.append('r')
                elif bottlenecks[k] == 'c':
                    value.append(Compute_Latency[k])
                    color.append('g')
                elif bottlenecks[k] == 'n':
                    value.append(Network_Latency[k])
                    color.append('b')
            
            for k in range(1, num_kernel):
                cumulative_value.append(value[k-1] + cumulative_value[k-1])
            
            for k in range(num_kernel):
                ax[i, j].bar(' ', value[k], bottom=cumulative_value[k], color=color[k])
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
    
                if i == 0 and j == 0:
                    ax[i, j].set_ylabel('GPU', fontsize=40)
                    
                if i == 1 and j == 0:
                    ax[i, j].set_ylabel('TPU', fontsize=40)
                    
                if i == 2 and j == 0:
                    ax[i, j].set_ylabel('RDU', fontsize=40)
                    
                if i == 3 and j == 0:
                    ax[i, j].set_ylabel('WSE', fontsize=40)
                    
                if i == 3 and j == 0:
                    ax[i, j].set_xlabel('2D Torus', fontsize=40, rotation=15)
                    
                if i == 3 and j == 1:
                    ax[i, j].set_xlabel('Dragonfly', fontsize=40, rotation=15)
                    
                if i == 3 and j == 2:
                    ax[i, j].set_xlabel('3D Torus', fontsize=40, rotation=15)
                    
                if i == 3 and j == 3:
                    ax[i, j].set_xlabel('DGX-1', fontsize=40, rotation=15)
                    
                if i == 3 and j == 4:
                    ax[i, j].set_xlabel('DGX-2', fontsize=40, rotation=15)
    
    plt.savefig('DLRM_'+combinations[m]+'.pdf', format="pdf", bbox_inches='tight')
    
            