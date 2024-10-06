import os

names = ['1', '2', '4', '8', '16', '32', '64', '128', '256']

for a in names:
    os.system('./run.sh LLM/Compare_w_existing/rail_only/16384/'+a+' > LLM/Compare_w_existing/rail_only/16384/'+a+'/log.txt')


    
