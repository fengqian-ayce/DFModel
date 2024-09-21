import copy
import os



topology = ['2d_torus', 'dragonfly', '3d_torus', 'dgx_1', 'dgx_2']
combination = ['ddr_pcie', 'ddr_nvlink', 'hbm_pcie', 'hbm_nvlink']

for c in combination:
    for t in topology:
        f = open(t+'_'+c+'/user_input/setup.txt', 'r')
        lines = f.readlines()
        mylines = copy.deepcopy(lines)
        f.close()
        
        for i in range(len(lines)):
            if lines[i].startswith('	perfect_overlap:'):
                mylines[i] = '	perfect_overlap: false\n'
        
        f = open(t+'_'+c+'/user_input/setup.txt', 'w')
        f.writelines(mylines)
        f.close()
            
            