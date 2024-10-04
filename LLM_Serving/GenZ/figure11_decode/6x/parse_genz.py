l = 96
compute = []
communication = []
attn_memory = 0
total_memory = 0

f = open('log.txt', 'r')


cnt = 0
lines = f.readlines()
for line in lines:
     if line.startswith('Network_Latency['):
          communication.append(float(line.split()[-1]))

     if line.startswith('Compute_Memory_Latency['):
          compute.append(float(line.split()[-1]))

     if line.startswith('memory_size'):
          if 5 <= cnt <= 10:
               attn_memory += float(line.split()[-1])
          total_memory += float(line.split()[-1])
          cnt += 1


f.close()

attn = (attn_memory / total_memory) * sum(compute) * l / 1e6
communication = sum(communication) * l / 1e6 / 2
gemm = (1 - attn_memory / total_memory) * sum(compute) * l / 1e6



print(gemm)
print(communication)
print(attn)


