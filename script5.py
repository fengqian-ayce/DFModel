import os
import multiprocessing


# aaa = ['regular_fft/256K', 'regular_fft/512K', 'regular_fft/1M', \
#        'gemm_fft/256K', 'gemm_fft/512K', 'gemm_fft/1M', \
#        'vector_fft/256K', 'vector_fft/512K', 'vector_fft/1M']

# bbb = ['regular_fft/s_sram_l_dram', 'regular_fft/l_sram_l_dram', \
#        'gemm_fft/s_sram_l_dram', 'gemm_fft/l_sram_l_dram', \
#        'vector_fft/s_sram_l_dram', 'vector_fft/l_sram_l_dram']

# aaa = ['attn/256K', 'attn/512K', 'attn/1M']


bbb = ['vector_fft/s_sram_s_dram', 'vector_fft/l_sram_s_dram']

# for n in aaa:
#     os.system('./run.sh SSM/'+n+' > SSM/'+n+'/log.txt')

for n in bbb:
    os.system('./run.sh SSM/Hyena/sweep_SRAM/'+n+' > SSM/Hyena/sweep_SRAM/'+n+'/log.txt')
