import os
import multiprocessing

params = ['10240', '12288', '16384', '20480', '25600']
  
def run(a): 
    os.system('./run.sh LLM/Compare_w_existing/GPU/Megatron_LM/'+a+'/ > LLM/Compare_w_existing/GPU/Megatron_LM/'+a+'/log.txt')

programs = []
for x in params:
    p = multiprocessing.Process(target=run, args=(x,))
    programs.append(p)
    p.start()
    
for program in programs:
    program.join()
    
    