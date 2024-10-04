import os

names = ['1x', '0.5x', '0.25x', '0.125x', '0.05x']

for a in names:
    os.system('./run.sh LLM_Serving/GenZ/figure13_decode/'+a+' > LLM_Serving/GenZ/figure13_decode/'+a+'/log.txt')
