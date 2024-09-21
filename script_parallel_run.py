import pandas as pd
import os
import multiprocessing
  
def run(n):
    os.system('./run.sh cloud/FFT_1K/'+n+' > cloud/FFT_1K/'+n+'/log.txt')


array = ['aws', 'azure', 'gcp']
for cloud in array:
    df = pd.read_excel('./cloud/FFT_1K/cloud.xlsx', cloud)
    torun = []
    for i in range(len(df['name'])):
        name = str(df['name'][i])
        gpu = str(df['gpu'][i])
        gpu_type = str(df['type'][i])
        
        name = cloud+'_'+name+'_'+gpu+'_'+gpu_type
        torun.append(name)
        
    programs = []
    for n in torun:
        p = multiprocessing.Process(target=run, args=(n, )) 
        programs.append(p)
        p.start() 
        
    for program in programs:
        program.join()

    