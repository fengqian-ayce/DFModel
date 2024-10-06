import os

name = ['256', '512', '1024', '2048', '4096']

for n in name:
    f = open(n+'/log.txt', 'r')
    lines = f.readlines()
    util = 0
    gflops = 0
    for line in lines:
        if line.startswith('System FLOPS Utilization'):
            util = float(line.split()[-1])
        if line.startswith('Per-Accelerator Throughput (GFLOPS)'):
            gflops = float(line.split()[-1])    
    print(util * gflops * float(n))    
    f.close()
    
    