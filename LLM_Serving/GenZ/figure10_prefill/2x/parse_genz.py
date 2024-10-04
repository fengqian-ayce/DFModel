l = 96
compute = []
communication = []
attn_flop = 0
total_flop = 0

f = open('log.txt', 'r')


cnt = 0
lines = f.readlines()
for line in lines:
     if line.startswith('Network_Latency['):
          communication.append(float(line.split()[-1]))

     if line.startswith('Compute_Memory_Latency['):
          compute.append(float(line.split()[-1]))

     if line.startswith('SIMD') or line.startswith('SYSTOLIC'):
          if 5 <= cnt <= 8:
               attn_flop += float(line.split()[-1])
          total_flop += float(line.split()[-1])
          cnt += 1


f.close()

attn = (attn_flop / total_flop) * sum(compute) * l / 1e6
communication = sum(communication) * l / 1e6
gemm = (1 - attn_flop / total_flop) * sum(compute) * l / 1e6


print(attn)
print(communication)
print(gemm)


