import os

accelerator = ['h100', 'tpu', 'sn30', 'wse-2']
name = ['32M', '1B', '32B', '1T']

for a in accelerator:
    for n in name:
        os.system('./run.sh FFT/DSE/scalability/'+a+'/multi_FFT/'+n+'/ > FFT/DSE/scalability/'+a+'/multi_FFT/'+n+'/log.txt')
        
        os.system('./run.sh FFT/DSE/scalability/'+a+'/one_FFT/'+n+'/ > FFT/DSE/scalability/'+a+'/one_FFT/'+n+'/log.txt')
    
    