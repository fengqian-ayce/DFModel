import os

name = ['s_sram_s_dram', \
        's_sram_m_dram', \
        's_sram_l_dram', \
        'm_sram_s_dram', \
        'm_sram_m_dram', \
        'm_sram_l_dram', \
        'l_sram_s_dram', \
        'l_sram_m_dram', \
        'l_sram_l_dram']

for n in name:
    f = open(n+'/log.txt', 'r')
    lines = f.readlines()
    for line in lines:
        if line.startswith('System FLOPS Utilization'):
            util = float(line.split()[-1])
            print(util)
    f.close()
