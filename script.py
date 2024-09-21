import os



accelerator = ['h100', 'tpu', 'sn30', 'wse-2']
name = \
['2d_torus_ddr_pcie', '2d_torus_ddr_nvlink', '2d_torus_hbm_pcie', '2d_torus_hbm_nvlink', \
'dragonfly_ddr_pcie', 'dragonfly_ddr_nvlink', 'dragonfly_hbm_pcie', 'dragonfly_hbm_nvlink', \
'3d_torus_ddr_pcie', '3d_torus_ddr_nvlink', '3d_torus_hbm_pcie', '3d_torus_hbm_nvlink', \
'dgx_1_ddr_pcie', 'dgx_1_ddr_nvlink', 'dgx_1_hbm_pcie', 'dgx_1_hbm_nvlink', \
'dgx_2_ddr_pcie', 'dgx_2_ddr_nvlink', 'dgx_2_hbm_pcie', 'dgx_2_hbm_nvlink'\
]

for a in accelerator:
    for n in name:
        os.system('./run.sh HPL/DSE/system/nonoverlap/'+a+'/'+n+' > HPL/DSE/system/nonoverlap/'+a+'/'+n+'/log.txt')
    
    