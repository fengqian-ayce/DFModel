import os

name = ['10240', '12288', '16384', '20480']

for n in name:
    f = open(n+'/log.txt', 'r')
    lines = f.readlines()
    util = 0
    gflops = 0
    for line in lines:
        if line.startswith('System FLOPS Utilization'):
            util = float(line.split()[-1]) 
    print(util)    
    f.close()
    