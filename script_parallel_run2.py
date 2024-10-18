import pandas as pd
import os
import multiprocessing
  
def run_2D_DF(i):
    os.system('./run.sh LLM_Serving/3D_memory/2D_DF/0.'+str(i)+'/ > LLM_Serving/3D_memory/2D_DF/0.'+str(i)+'/log.txt')

def run_2D_KBK(i):
    os.system('./run.sh LLM_Serving/3D_memory/2D_KBK/0.'+str(i)+'/ > LLM_Serving/3D_memory/2D_KBK/0.'+str(i)+'/log.txt')

def run_3D_DF(i):
    os.system('./run.sh LLM_Serving/3D_memory/3D_DF/0.'+str(i)+'/ > LLM_Serving/3D_memory/3D_DF/0.'+str(i)+'/log.txt')

def run_3D_KBK(i):
    os.system('./run.sh LLM_Serving/3D_memory/3D_KBK/0.'+str(i)+'/ > LLM_Serving/3D_memory/3D_KBK/0.'+str(i)+'/log.txt')





for i in range(1, 10):
    run_2D_KBK(i)



# programs = []
# for i in range(1, 10):
#     p = multiprocessing.Process(target=run_2D, args=(i, )) 
#     programs.append(p)
#     p.start() 
    
# for program in programs:
#     program.join()





# programs = []
# for i in range(1, 10):
#     p = multiprocessing.Process(target=run_3D, args=(i, )) 
#     programs.append(p)
#     p.start() 
    
# for program in programs:
#     program.join()
