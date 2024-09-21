import os



accelerator = ['h100', 'tpu', 'sn30', 'wse-2']
topology = ['2d_torus', 'dragonfly', '3d_torus', 'dgx_1', 'dgx_2']
combination = ['ddr_pcie', 'ddr_nvlink', 'hbm_pcie', 'hbm_nvlink']

util = []
total_cost = []
total_power = []

for c in combination:
    for a in accelerator:
        for t in topology:
            f = open(a+'/'+t+'_'+c+'/log.txt', 'r')
            lines = f.readlines()
            for line in lines:
                if line.startswith('util'):
                    util.append(line.split()[-1])
                    print(c, a, t)
            f.close()

for c in combination:
    for a in accelerator:
        for t in topology:
            f = open(a+'/'+t+'_'+c+'/log.txt', 'r')
            lines = f.readlines()
            for line in lines:
                if line.startswith('total_cost'):
                    total_cost.append(line.split()[-1])
                    break
            f.close()


for c in combination:
    for a in accelerator:
        for t in topology:
            f = open(a+'/'+t+'_'+c+'/log.txt', 'r')
            lines = f.readlines()
            for line in lines:
                if line.startswith('total_power'):
                    total_power.append(line.split()[-1])
                    break
            f.close()

        

for i in range(int(len(util)/5)):
    for j in range(5):
        print(util[i*5+j], end=' ')
    print()    

print()

for i in range(int(len(util)/5)):
    for j in range(5):
        print(total_cost[i*5+j], end=' ')
    print() 

print()


for i in range(int(len(util)/5)):
    for j in range(5):
        print(total_power[i*5+j], end=' ')
    print()        