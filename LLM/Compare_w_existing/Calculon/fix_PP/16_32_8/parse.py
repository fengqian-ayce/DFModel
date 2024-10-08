import sys
import pprint

f = open('log.txt', 'r')

Setup_Latency = []
Memory_Latency = []
Compute_Latency = []
Network_Latency_ALL_REDUCE = []
Network_Latency_ALL_REDUCE_PERIODIC = []
pipeline_factor = 0
p2p_latency = 0
Per_Config_II = []
TP = 0
PP = 0
DP = 0

batch = 3072
num_kernel = 41
num_layer = 128

lines = f.readlines()
for line in lines:
    if line.startswith('TP '):
        TP = float(line.split()[-1])
    if line.startswith('PP '):
        PP = float(line.split()[-1])
    if line.startswith('DP '):
        DP = float(line.split()[-1])
    if line.startswith('Memory_Latency['):
        Memory_Latency.append(float(line.split()[-1]))
    if line.startswith('Compute_Latency['):
        Compute_Latency.append(float(line.split()[-1]))
    if line.startswith('Network_Latency_ALL_REDUCE_node['):
        Network_Latency_ALL_REDUCE.append(float(line.split()[-1]))
    if line.startswith('Network_Latency_ALL_REDUCE_PERIODIC_node['):
        Network_Latency_ALL_REDUCE_PERIODIC.append(float(line.split()[-1]))
    if line.startswith('pipeline_factor'):
        pipeline_factor = float(line.split()[-1])
    if line.startswith('p2p_latency'):
        p2p_latency = float(line.split()[-1])


fwd_pass = sum(Memory_Latency[:18]) + sum(Compute_Latency[:18])
bwd_pass = sum(Memory_Latency[18:]) + sum(Compute_Latency[18:])
tp_comm = sum(Network_Latency_ALL_REDUCE)
dp_comm = sum(Network_Latency_ALL_REDUCE_PERIODIC)
pp_comm = p2p_latency
pp_bubble = (fwd_pass + bwd_pass + tp_comm + dp_comm) * (pipeline_factor - 1)

II = fwd_pass + bwd_pass + tp_comm + dp_comm + pp_bubble
II *= num_layer / PP
sec_per_batch = batch / (1e9 * DP / (II)) + pp_comm / 1e9

print(num_layer / PP * batch / 1e9 / DP * fwd_pass)
print(num_layer / PP * batch / 1e9 / DP * bwd_pass)
print(num_layer / PP * batch / 1e9 / DP * pp_bubble)
print(num_layer / PP * batch / 1e9 / DP * tp_comm)
print(pp_comm / 1e9)
print(num_layer / PP * batch / 1e9 / DP * dp_comm)
print(sec_per_batch)





# get memory usage
# weight = 0
# weight_grad = 0
# activation = 0
# activation_grad = 0

# for line in lines:
#     if line.startswith('weight '):
#         weight = float(line.split()[-1]) / 1024**3
#     if line.startswith('weight_grad '):
#         weight_grad = float(line.split()[-1]) / 1024**3
#     if line.startswith('activation '):
#         activation = float(line.split()[-1]) / 1024**3
#     if line.startswith('activation_grad '):
#         activation_grad = float(line.split()[-1]) / 1024**3
        
# print(weight)
# print(activation)
# print(activation_grad)
# print(weight_grad)
# print(weight + activation + activation_grad + weight_grad)



