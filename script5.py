import os
import multiprocessing

# accelerator = ['h100', 'tpu', 'sn30', 'wse-2']
# name = ['2d_torus_ddr_pcie', '2d_torus_ddr_nvlink', '2d_torus_hbm_pcie', '2d_torus_hbm_nvlink', \
# 'dragonfly_ddr_pcie', 'dragonfly_ddr_nvlink', 'dragonfly_hbm_pcie', 'dragonfly_hbm_nvlink', \
# '3d_torus_ddr_pcie', '3d_torus_ddr_nvlink', '3d_torus_hbm_pcie', '3d_torus_hbm_nvlink', \
# 'dgx_1_ddr_pcie', 'dgx_1_ddr_nvlink', 'dgx_1_hbm_pcie', 'dgx_1_hbm_nvlink', \
# 'dgx_2_ddr_pcie', 'dgx_2_ddr_nvlink', 'dgx_2_hbm_pcie', 'dgx_2_hbm_nvlink'\
# ]


names = ['1', '2', '4', '8', '16', '32', '64', '128', '256']

def run(n): 
    os.system('./run.sh rail_only/1T/16384/'+n+'/ > rail_only/1T/16384/'+n+'/log.txt')

programs = []
for n in names:
    p = multiprocessing.Process(target=run, args=(n, ))
    programs.append(p)
    p.start()

for program in programs:
    program.join()
    
    