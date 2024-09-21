import os
import multiprocessing
import time



name = ['s_sram_s_dram', \
        's_sram_m_dram', \
        's_sram_l_dram', \
        'm_sram_s_dram', \
        'm_sram_m_dram', \
        'm_sram_l_dram', \
        'l_sram_s_dram', \
        'l_sram_m_dram', \
        'l_sram_l_dram']

def run_kbk(n):
    os.system('./run.sh LLM/sram_dram/KBK/'+n+'/ > LLM/sram_dram/KBK/'+n+'/log.txt')

def run_df(n):
    os.system('./run.sh LLM/sram_dram/DF/'+n+'/ > LLM/sram_dram/DF/'+n+'/log.txt')



start_time = time.time()

for n in name:
    run_kbk(n)

for n in name:
    run_df(n)

end_time = time.time()


execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")


