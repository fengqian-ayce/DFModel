import h5py
import argparse
from highspy import Highs  # Assumes HiGHS Python API
import numpy as np
import setup_pb2
import pprint
from enum import Enum
from google.protobuf import text_format
import pydot
import copy
import sys
import math
import time
import os

large_number = 999999999999

# user pass in
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help='pls pass in the folder name under the current directory (named after the DL model)', required=True)
args = parser.parse_args()
name = args.name



# read in pd file
dse = setup_pb2.DSE()
with open('./'+name+'/'+'dse_sharded.pb', "rb") as file:
    dse.ParseFromString(file.read())
    
    


# get kernels
class Dim(Enum):
    DIM_PLACEHOLDER = 0
    OUTER_DIM = 1
    M_DIM = 2
    K_DIM = 3
    N_DIM = 4
    NO_DIM = 5

class KernelType(Enum):
    NO_Type = 0
    SYSTOLIC = 1
    SIMD = 2

class Communication(Enum):
    NO_COMMUNICATION = 0
    ALL_REDUCE = 1
    ALL_TO_ALL = 2
    ALL_GATHER = 3
    ALL_REDUCE_PERIODIC = 4
    POINT_TO_POINT = 5
    BROADCAST = 6
    
class Execution_Style(Enum):
    NO_Execution_Style = 0
    DATAFLOW = 1
    KERNEL_BY_KERNEL = 2

class Overlap_Style(Enum):
    NO_OVERLAP_Style = 0
    PERFECT_OVERLAP = 1
    NO_OVERLAP = 2
    OVERLAP_COMPUTE_MEMORY = 3

class FWD_BWD(Enum):
    Placeholder = 0
    FWD = 1
    BWD = 2
    
class BasicTopology(Enum):
    NO_BASICTOPOLOGY = 0
    R = 1
    FC = 2
    SW = 3

    

kernel_id = []
kernel_name = []   
kernel_type = []
configs = []
fwd_bwd = []
topological_number = []

M = []
K = []
N = []
weight_tensor_size = []

sharding = []
memory_size = []

node_communication_type = []
node_communication_size = []

node_communication_type_2 = []
node_communication_size_2 = []


use_effective_stage = []
num_input = []
num_input_set = []
skip_weight = []
node_type = []
tiling = []
sram_extra = []
dram_extra = []
node_dict = {} # map kernel id to index in list
i = 0
for kernel in dse.dataflow_graph.kernels:
    kernel_id.append(kernel.id)
    kernel_name.append(kernel.name)
    kernel_type.append(kernel.type)
    fwd_bwd.append(kernel.fwd_bwd)
    configs.append(kernel.config)
    topological_number.append(kernel.topological_number)
    
    if kernel.WhichOneof('kernel_variant') == 'gemm_input1_weight':  
        M.append(kernel.gemm_input1_weight.outer * kernel.gemm_input1_weight.M)
        K.append(kernel.gemm_input1_weight.K)
        N.append(kernel.gemm_input1_weight.N)
        
        weight_tensor_size.append(kernel.gemm_input1_weight.weight_tensor_size)
        
        sharding.append(kernel.gemm_input1_weight.sharding)
        node_communication_type.append(kernel.gemm_input1_weight.communication_type)
        node_communication_size.append(kernel.gemm_input1_weight.communication_size)
        
        node_communication_type_2.append(kernel.gemm_input1_weight.communication_type_2)
        node_communication_size_2.append(kernel.gemm_input1_weight.communication_size_2)
        
        tiling.append(kernel.gemm_input1_weight.tiling)
        
        memory_size.append(kernel.gemm_input1_weight.memory_size)
        

        sram_extra.append(kernel.gemm_input1_weight.sram_extra)
        dram_extra.append(kernel.gemm_input1_weight.dram_extra)

        node_type.append('gemm_input1_weight')

        skip_weight.append(kernel.gemm_input1_weight.skip_weight)
        use_effective_stage.append(kernel.gemm_input1_weight.use_effective_stage)
        
        if kernel.gemm_input1_weight.num_input == 0:
            num_input.append(1)
            num_input_set.append(False)
        else:
            num_input.append(kernel.gemm_input1_weight.num_input)
            num_input_set.append(True)
        

    elif kernel.WhichOneof('kernel_variant') == 'gemm_input1_input2':
        M.append(kernel.gemm_input1_input2.outer * kernel.gemm_input1_input2.M)
        K.append(kernel.gemm_input1_input2.K)
        N.append(kernel.gemm_input1_input2.N)
        
        weight_tensor_size.append(-1.0)
        
        sharding.append(kernel.gemm_input1_input2.sharding)
        node_communication_type.append(kernel.gemm_input1_input2.communication_type)
        node_communication_size.append(kernel.gemm_input1_input2.communication_size)
        
        node_communication_type_2.append(kernel.gemm_input1_input2.communication_type_2)
        node_communication_size_2.append(kernel.gemm_input1_input2.communication_size_2)
        
        tiling.append(kernel.gemm_input1_input2.tiling)
        
        memory_size.append(kernel.gemm_input1_input2.memory_size)
        

        sram_extra.append(kernel.gemm_input1_input2.sram_extra)
        dram_extra.append(kernel.gemm_input1_input2.dram_extra)


        node_type.append('gemm_input1_input2')
        
        skip_weight.append(False)
        use_effective_stage.append(False)

        if kernel.gemm_input1_input2.num_input == 0:
            num_input.append(1)
            num_input_set.append(False)
        else:
            num_input.append(kernel.gemm_input1_input2.num_input)
            num_input_set.append(True)

    elif kernel.WhichOneof('kernel_variant') == 'elementwise_input1':
        M.append(kernel.elementwise_input1.outer * kernel.elementwise_input1.M)
        K.append(1)
        N.append(kernel.elementwise_input1.N)
        
        weight_tensor_size.append(-1.0)
        
        sharding.append(kernel.elementwise_input1.sharding)
        node_communication_type.append(kernel.elementwise_input1.communication_type)
        node_communication_size.append(kernel.elementwise_input1.communication_size)
        
        node_communication_type_2.append(kernel.elementwise_input1.communication_type_2)
        node_communication_size_2.append(kernel.elementwise_input1.communication_size_2)
        
        tiling.append(kernel.elementwise_input1.tiling)
        
        memory_size.append(kernel.elementwise_input1.memory_size)
        

        sram_extra.append(kernel.elementwise_input1.sram_extra)
        dram_extra.append(kernel.elementwise_input1.dram_extra)


        node_type.append('elementwise_input1')
        
        skip_weight.append(False)
        use_effective_stage.append(False)

        if kernel.elementwise_input1.num_input == 0:
            num_input.append(1)
            num_input_set.append(False)
        else:
            num_input.append(kernel.elementwise_input1.num_input)
            num_input_set.append(True)


    elif kernel.WhichOneof('kernel_variant') == 'elementwise_input1_input2':
        M.append(kernel.elementwise_input1_input2.outer * kernel.elementwise_input1_input2.M)
        K.append(1)
        N.append(kernel.elementwise_input1_input2.N)
        
        weight_tensor_size.append(-1.0)
        
        sharding.append(kernel.elementwise_input1_input2.sharding)
        node_communication_type.append(kernel.elementwise_input1_input2.communication_type)
        node_communication_size.append(kernel.elementwise_input1_input2.communication_size)
        
        node_communication_type_2.append(kernel.elementwise_input1_input2.communication_type_2)
        node_communication_size_2.append(kernel.elementwise_input1_input2.communication_size_2)
        
        tiling.append(kernel.elementwise_input1_input2.tiling)
        
        memory_size.append(kernel.elementwise_input1_input2.memory_size)
        

        sram_extra.append(kernel.elementwise_input1_input2.sram_extra)
        dram_extra.append(kernel.elementwise_input1_input2.dram_extra)


        node_type.append('elementwise_input1_input2')
        
        skip_weight.append(False)
        use_effective_stage.append(False)

        if kernel.elementwise_input1_input2.num_input == 0:
            num_input.append(1)
            num_input_set.append(False)
        else:
            num_input.append(kernel.elementwise_input1_input2.num_input)
            num_input_set.append(True)

    else:
        raise Exception('Wrong!')
    
    node_dict[kernel.id] = i
    i += 1

num_node = len(kernel_name)
memory_size = np.array(memory_size)












# get edges
startIdx = []
endIdx = []
depth = []
tensor_size = []
lane_stage_type = []
edge_communication_type = []
edge_communication_type_2 = []
zero_out = []
for connection in dse.dataflow_graph.connections:
    startIdx.append(connection.startIdx)
    endIdx.append(connection.endIdx)
    depth.append(connection.buffer_depth)
    tensor_size.append(connection.tensor_size)
    lane_stage_type.append(connection.lane_stage_type)
    edge_communication_type.append(connection.communication_type)
    edge_communication_type_2.append(connection.communication_type_2)
    zero_out.append(connection.zero_out)

num_edge = len(startIdx)





print(node_communication_type_2)
print(edge_communication_type_2)
print('zzzzzzzzzzzzzzzzzzzzz')











# get weights
weight_dict = {} # index in weights to node id
cnt = 0
for i in range(len(weight_tensor_size)):
    if weight_tensor_size[i] != -1:
        weight_dict[cnt] = kernel_id[i]
        cnt += 1
num_weight = len(weight_dict.keys())











        
        

if dse.execution.WhichOneof('workload_variant') == 'llm':
    hidden_dim = dse.execution.llm.hidden_dim
    head_dim = dse.execution.llm.head_dim
    num_head = dse.execution.llm.num_head
    seq_len = dse.execution.llm.seq_len
    
    global_batch_size = dse.execution.llm.global_batch_size
    micro_batch_size = dse.execution.llm.micro_batch_size
    
    
    if dse.execution.llm.global_batch_size == 0:
        raise Exception('Wrong!')

elif dse.execution.WhichOneof('workload_variant') == 'dlrm':
    num_table = dse.execution.dlrm.num_table
    emb_dim = dse.execution.dlrm.emb_dim
    row = dse.execution.dlrm.row
    
    global_batch_size = dse.execution.dlrm.global_batch_size
    micro_batch_size = 1
    num_layer = 1
    
    if dse.execution.dlrm.global_batch_size == 0:
        raise Exception('Wrong!')

elif dse.execution.WhichOneof('workload_variant') == 'hpl':
    n = dse.execution.hpl.n
    b = dse.execution.hpl.b
    
    global_batch_size = 1
    micro_batch_size = 1
    num_layer = 1
    
elif dse.execution.WhichOneof('workload_variant') == 'fft':
    length = dse.execution.fft.length
    
    global_batch_size = 1
    micro_batch_size = 1
    num_layer = 1

else:
    raise Exception('Wrong!')


word = dse.execution.word
if word == 0:
    raise Exception('Wrong!')





last_fwd_kernel = -1
first_bwd_kernel = -1
for i in range(num_node):
    if fwd_bwd[i] == FWD_BWD.FWD.value:
        last_fwd_kernel = i

for i in range(num_node):       
    if fwd_bwd[i] == FWD_BWD.BWD.value:
        first_bwd_kernel = i
        break



link_unit_price = dse.cost.link_unit_price
switch_unit_price = dse.cost.switch_unit_price
dram_unit_price = dse.cost.dram_unit_price
accelerator_price = dse.cost.accelerator_price
link_unit_power_x = dse.cost.link_unit_power_x
link_unit_power_y = dse.cost.link_unit_power_y
link_unit_power_z = dse.cost.link_unit_power_z
switch_unit_power = dse.cost.switch_unit_power
dram_unit_power = dse.cost.dram_unit_power
accelerator_power = dse.cost.accelerator_power

if dse.execution.WhichOneof('workload_variant') == 'llm':
    if num_head * head_dim != hidden_dim:
        raise Exception('Wrong!')
    
    if first_bwd_kernel == -1: # inference
        Intermediate = 0
    else: # training
        Intermediate = 2 * hidden_dim * seq_len * word

elif dse.execution.WhichOneof('workload_variant') == 'dlrm' or dse.execution.WhichOneof('workload_variant') == 'hpl' or dse.execution.WhichOneof('workload_variant') == 'fft':
    Intermediate = 0
    
else:
    raise Exception('Wrong!')



    







# get system info
Core = dse.system.accelerator.core
LaneWidth = dse.system.accelerator.systolic_width
StageWidth = dse.system.accelerator.systolic_height
Freq = dse.system.accelerator.freq
num_chip = dse.system.num_chip
GFLOPS = 2*LaneWidth*StageWidth*Core*Freq

if dse.execution.WhichOneof('workload_variant') == 'dlrm':
    Num_Chips_Per_Copy = num_chip / dse.execution.dlrm.num_copy

elif dse.execution.WhichOneof('workload_variant') == 'fft':
    Num_Chips_Per_Copy = num_chip / dse.execution.fft.num_copy




if dse.system.WhichOneof('topology_variant') == 'sw': # 1D SW
    topology = [BasicTopology.SW.value]
    link_bw = [dse.system.sw.link_bw_x]
    dimension = [dse.system.sw.x]
    par = [dse.system.sw.par_x]

    if dse.execution.WhichOneof('workload_variant') == 'dlrm' or dse.execution.WhichOneof('workload_variant') == 'fft':
        a2a_bw_factor = [dse.system.sw.x]
        a2a_msg_factor = [Num_Chips_Per_Copy**2 / 4]
        
elif dse.system.WhichOneof('topology_variant') == 'fc': # 1D FC
    topology = [BasicTopology.FC.value]
    link_bw = [dse.system.fc.link_bw_x]
    dimension = [dse.system.fc.x]
    par = [dse.system.fc.par_x]
    
    if dse.execution.WhichOneof('workload_variant') == 'dlrm' or dse.execution.WhichOneof('workload_variant') == 'fft':
        a2a_bw_factor = [dse.system.fc.x**2 / 4]
        a2a_msg_factor = [Num_Chips_Per_Copy**2 / 4]
    
elif dse.system.WhichOneof('topology_variant') == 'r': # 1D Ring
    topology = [BasicTopology.R.value]
    link_bw = [dse.system.r.link_bw_x]
    dimension = [dse.system.r.x]    
    par = [dse.system.r.par_x]
    
    if dse.execution.WhichOneof('workload_variant') == 'dlrm' or dse.execution.WhichOneof('workload_variant') == 'fft':
        a2a_bw_factor = [2]
        a2a_msg_factor = [Num_Chips_Per_Copy**2 / 4]
    
elif dse.system.WhichOneof('topology_variant') == 'r_r': # 2D Torus
    topology = [BasicTopology.R.value, BasicTopology.R.value]
    link_bw = [dse.system.r_r.link_bw_x, dse.system.r_r.link_bw_y]
    dimension = [dse.system.r_r.x, dse.system.r_r.y]
    par = [dse.system.r_r.par_x, dse.system.r_r.par_y]
    
    if dse.execution.WhichOneof('workload_variant') == 'dlrm' or dse.execution.WhichOneof('workload_variant') == 'fft':
        a2a_bw_factor = [2*dse.system.r_r.x, 2*dse.system.r_r.y]
        a2a_msg_factor = [Num_Chips_Per_Copy**2 / 4, Num_Chips_Per_Copy**2 / 4]

elif dse.system.WhichOneof('topology_variant') == 'fc_fc': # 2D Dragonfly
    topology = [BasicTopology.FC.value, BasicTopology.FC.value]
    link_bw = [dse.system.fc_fc.link_bw_x, dse.system.fc_fc.link_bw_y]
    dimension = [dse.system.fc_fc.x, dse.system.fc_fc.y]
    par = [dse.system.fc_fc.par_x, dse.system.fc_fc.par_y]
    
    if dse.execution.WhichOneof('workload_variant') == 'dlrm' or dse.execution.WhichOneof('workload_variant') == 'fft':
        a2a_bw_factor = [dse.system.fc_fc.x**2 / 4, dse.system.fc_fc.y**2 / 4]
        a2a_msg_factor = [dse.system.fc_fc.x**2 / 4, Num_Chips_Per_Copy**2 / 4]
    
elif dse.system.WhichOneof('topology_variant') == 'r_fc': # 2D ZionEX
    topology = [BasicTopology.R.value, BasicTopology.FC.value]
    link_bw = [dse.system.r_fc.link_bw_x, dse.system.r_fc.link_bw_y]
    dimension = [dse.system.r_fc.x, dse.system.r_fc.y]
    par = [dse.system.r_fc.par_x, dse.system.r_fc.par_y]
    
    if dse.execution.WhichOneof('workload_variant') == 'dlrm' or dse.execution.WhichOneof('workload_variant') == 'fft':
        a2a_bw_factor = [2, dse.system.r_fc.y**2 / 4]
        a2a_msg_factor = [dse.system.r_fc.x**2 / 4, Num_Chips_Per_Copy**2 / 4]

elif dse.system.WhichOneof('topology_variant') == 'r_sw': # 2D DGX-1
    topology = [BasicTopology.R.value, BasicTopology.SW.value]
    link_bw = [dse.system.r_sw.link_bw_x, dse.system.r_sw.link_bw_y]
    dimension = [dse.system.r_sw.x, dse.system.r_sw.y]
    par = [dse.system.r_sw.par_x, dse.system.r_sw.par_y]
    
    if dse.execution.WhichOneof('workload_variant') == 'dlrm' or dse.execution.WhichOneof('workload_variant') == 'fft':
        a2a_bw_factor = [2, dse.system.r_sw.y]
        a2a_msg_factor = [dse.system.r_sw.x**2 / 4, Num_Chips_Per_Copy**2 / 4]
    
elif dse.system.WhichOneof('topology_variant') == 'sw_sw': # 2D DGX-2
    topology = [BasicTopology.SW.value, BasicTopology.SW.value]
    link_bw = [dse.system.sw_sw.link_bw_x, dse.system.sw_sw.link_bw_y]
    dimension = [dse.system.sw_sw.x, dse.system.sw_sw.y]
    par = [dse.system.sw_sw.par_x, dse.system.sw_sw.par_y]
    
    if dse.execution.WhichOneof('workload_variant') == 'dlrm' or dse.execution.WhichOneof('workload_variant') == 'fft':
        a2a_bw_factor = [dse.system.sw_sw.x, dse.system.sw_sw.y]
        a2a_msg_factor = [dse.system.sw_sw.x**2 / 4, Num_Chips_Per_Copy**2 / 4]
        
elif dse.system.WhichOneof('topology_variant') == 'r_r_r': # 3D Torus
    topology = [BasicTopology.R.value, BasicTopology.R.value, BasicTopology.R.value]
    link_bw = [dse.system.r_r_r.link_bw_x, dse.system.r_r_r.link_bw_y, dse.system.r_r_r.link_bw_z]
    dimension = [dse.system.r_r_r.x, dse.system.r_r_r.y, dse.system.r_r_r.z]
    par = [dse.system.r_r_r.par_x, dse.system.r_r_r.par_y, dse.system.r_r_r.par_z]
    
    if dse.execution.WhichOneof('workload_variant') == 'dlrm' or dse.execution.WhichOneof('workload_variant') == 'fft':
        a2a_bw_factor = [2*dse.system.r_r_r.x * dse.system.r_r_r.y, 2*dse.system.r_r_r.x * dse.system.r_r_r.z, 2*dse.system.r_r_r.y * dse.system.r_r_r.z]
        a2a_msg_factor = [Num_Chips_Per_Copy**2 / 4, Num_Chips_Per_Copy**2 / 4, Num_Chips_Per_Copy**2 / 4]

elif dse.system.WhichOneof('topology_variant') == 'r_sw_sw': # 3D DGX-1
    topology = [BasicTopology.R.value, BasicTopology.SW.value, BasicTopology.SW.value]
    link_bw = [dse.system.r_sw_sw.link_bw_x, dse.system.r_sw_sw.link_bw_y, dse.system.r_sw_sw.link_bw_z]
    dimension = [dse.system.r_sw_sw.x, dse.system.r_sw_sw.y, dse.system.r_sw_sw.z]
    par = [dse.system.r_sw_sw.par_x, dse.system.r_sw_sw.par_y, dse.system.r_sw_sw.par_z]
    
    if dse.execution.WhichOneof('workload_variant') == 'dlrm' or dse.execution.WhichOneof('workload_variant') == 'fft':
        a2a_bw_factor = [2, dse.system.r_sw_sw.y, dse.system.r_sw_sw.z]
        a2a_msg_factor = [dse.system.r_sw_sw.x**2 / 4, (dse.system.r_sw_sw.x*dse.system.r_sw_sw.y)**2 / 4, Num_Chips_Per_Copy**2 / 4]
    
elif dse.system.WhichOneof('topology_variant') == 'sw_sw_sw': # 3D DGX-2
    topology = [BasicTopology.SW.value, BasicTopology.SW.value, BasicTopology.SW.value]
    link_bw = [dse.system.sw_sw_sw.link_bw_x, dse.system.sw_sw_sw.link_bw_y, dse.system.sw_sw_sw.link_bw_z]
    dimension = [dse.system.sw_sw_sw.x, dse.system.sw_sw_sw.y, dse.system.sw_sw_sw.z]
    par = [dse.system.sw_sw_sw.par_x, dse.system.sw_sw_sw.par_y, dse.system.sw_sw_sw.par_z]
    
    if dse.execution.WhichOneof('workload_variant') == 'dlrm' or dse.execution.WhichOneof('workload_variant') == 'fft':
        a2a_bw_factor = [dse.system.sw_sw_sw.x, dse.system.sw_sw_sw.y, dse.system.sw_sw_sw.z]
        a2a_msg_factor = [dse.system.sw_sw_sw.x**2 / 4, (dse.system.sw_sw_sw.x*dse.system.sw_sw_sw.y)**2 / 4, Num_Chips_Per_Copy**2 / 4]

elif dse.system.WhichOneof('topology_variant') == 'r_r_r_r': # 4D torus
    topology = [BasicTopology.R.value, BasicTopology.R.value, BasicTopology.R.value, BasicTopology.R.value]
    link_bw = [dse.system.r_r_r_r.link_bw_w, dse.system.r_r_r_r.link_bw_x, dse.system.r_r_r_r.link_bw_y, dse.system.r_r_r_r.link_bw_z]
    dimension = [dse.system.r_r_r_r.w, dse.system.r_r_r_r.x, dse.system.r_r_r_r.y, dse.system.r_r_r_r.z]
    par = [dse.system.r_r_r_r.par_w, dse.system.r_r_r_r.par_x, dse.system.r_r_r_r.par_y, dse.system.r_r_r_r.par_z]

else:
    raise Exception('Wrong!')


DRAM_Cap = dse.system.memory.dram_cap
if DRAM_Cap == 0:
    DRAM_Cap = sys.maxsize

if dse.execution.WhichOneof('workload_variant') == 'dlrm':
    table_size = num_table*row*emb_dim*word
    print('total table size', table_size)




start = time.time()





model = Highs()
# HiGHS does not require setting NonConvex; adjust thread, gap, and time parameters as needed:
# model.set_parameter('threads', dse.gurobi.thread if dse.gurobi.thread != 0 else os.cpu_count())
# model.set_parameter('mip_gap', 0.001 if dse.gurobi.gap==0 else dse.gurobi.gap)
# model.set_parameter('time_limit', dse.gurobi.time) if dse.gurobi.time != 0 else None





# SP/TP/PP/DP
SP = model.add_variable(name='SP', vtype='integer', lb=1)
TP = model.add_variable(name='TP', vtype='integer', lb=1)
PP = model.add_variable(name='PP', vtype='integer', lb=1)
DP = model.add_variable(name='DP', vtype='integer', lb=1)

num_copy = model.add_variable(name='num_copy', vtype='integer', lb=1)
num_chips_per_copy = model.add_variable(name='num_chips_per_copy', vtype='integer', lb=1)
model.add_constraint(num_copy * num_chips_per_copy == num_chip)


if dse.execution.WhichOneof('workload_variant') == 'fft':
    Shape = model.add_matrix_variable(len(topology), name='Shape', vtype='integer', lb=1)
    
    model.add_constraint(TP == 1)
    model.add_constraint(PP == 1)
    model.add_constraint(DP == 1)

    Link_BW = model.add_matrix_variable(len(topology), name='Link_BW', vtype='continuous', lb=0)
    for i in range(len(topology)):
        model.add_constraint(Link_BW[i] == link_bw[i])
        
    FFT_DRAM_SIZE = model.add_variable(name='FFT_DRAM_SIZE', vtype='continuous', lb=0)
    side = int(length**0.5)
    model.add_constraint(FFT_DRAM_SIZE * num_chips_per_copy >= length * word)
    
    if len(topology) == 1:
        model.add_constraint(Shape[0] == num_chips_per_copy)
        model.add_constraint(Shape[0] == dimension[0])
        
    elif len(topology) == 2:
        model.add_constraint(Shape[0] * Shape[1] == num_chips_per_copy)
        model.add_constraint(Shape[0] == dimension[0])  
        model.add_constraint(Shape[1] == dimension[1])

    elif len(topology) == 3:        
        tmp = model.add_variable(vtype='integer', lb=1)
        model.add_constraint(Shape[0] * Shape[1] == tmp)
        model.add_constraint(tmp * Shape[2] == num_chips_per_copy)
        model.add_constraint(Shape[0] == dimension[0])  
        model.add_constraint(Shape[1] == dimension[1])
        model.add_constraint(Shape[2] == dimension[2])        
        
    else:
        raise Exception('False')
    model.add_constraint(num_copy == dse.execution.fft.num_copy)
    
    
elif dse.execution.WhichOneof('workload_variant') == 'hpl':
    Shape = model.add_matrix_variable(len(topology), name='Shape', vtype='integer', lb=0)
    
    model.add_constraint(TP == 1)
    model.add_constraint(PP == 1)
    model.add_constraint(DP == 1)

    Link_BW = model.add_matrix_variable(len(topology), name='Link_BW', vtype='continuous', lb=0)
    for i in range(len(topology)):
        model.add_constraint(Link_BW[i] == link_bw[i])
    
    HPL_DRAM_SIZE = model.add_variable(name='HPL_DRAM_SIZE', vtype='continuous', lb=0)
    model.add_constraint(HPL_DRAM_SIZE * num_chips_per_copy >= n * n * word)
    
    HPL_sharding_factor_X = model.add_variable(name='HPL_sharding_factor_X', vtype='integer', lb=1)
    HPL_sharding_factor_Y = model.add_variable(name='HPL_sharding_factor_Y', vtype='integer', lb=1)
    
    if len(topology) == 1:
        model.add_constraint(HPL_sharding_factor_X == 1)
        model.add_constraint(HPL_sharding_factor_Y == Shape[0])
        model.add_constraint(Shape[0] == num_chips_per_copy)
        
        model.add_constraint(Shape[0] == dimension[0])
        
    elif len(topology) == 2:
        model.add_constraint(HPL_sharding_factor_X == Shape[0])
        model.add_constraint(HPL_sharding_factor_Y == Shape[1])
        model.add_constraint(Shape[0] * Shape[1] == num_chips_per_copy)
        
        model.add_constraint(Shape[0] == dimension[0])
        model.add_constraint(Shape[1] == dimension[1])
            
    elif len(topology) == 3:
        model.add_constraint(HPL_sharding_factor_X == Shape[0] * Shape[1])
        model.add_constraint(HPL_sharding_factor_Y == Shape[2])
        tmp = model.add_variable(vtype='integer', lb=1)
        model.add_constraint(Shape[0] * Shape[1] == tmp)
        model.add_constraint(tmp * Shape[2] == num_chips_per_copy)
        
        model.add_constraint(Shape[0] == dimension[0])
        model.add_constraint(Shape[1] == dimension[1])
        model.add_constraint(Shape[2] == dimension[2])        

    else:
        raise Exception('Wrong')
    model.add_constraint(num_copy == dse.execution.hpl.num_copy)
    
    
    
elif dse.execution.WhichOneof('workload_variant') == 'dlrm':
    sharded_table_size = model.add_variable(name='sharded_table', vtype='continuous', lb=0)
    model.add_constraint(sharded_table_size * num_chips_per_copy >= table_size)

    Shape = model.add_matrix_variable(len(topology), name='Shape', vtype='integer', lb=0)
    if len(topology) == 1:
        model.add_constraint(Shape[0] == dimension[0])
        model.add_constraint(Shape[0] == num_chips_per_copy)
        
    elif len(topology) == 2:
        model.add_constraint(Shape[0] == dimension[0])
        model.add_constraint(Shape[1] == dimension[1])
        model.add_constraint(Shape[0] * Shape[1] == num_chips_per_copy)
        
    elif len(topology) == 3:
        model.add_constraint(Shape[0] == dimension[0])
        model.add_constraint(Shape[1] == dimension[1])
        model.add_constraint(Shape[2] == dimension[2])     
        tmp = model.add_variable(vtype='integer', lb=1)
        model.add_constraint(Shape[0] * Shape[1] == tmp)
        model.add_constraint(tmp * Shape[2] == num_chips_per_copy)
    
    else:
        raise Exception('Wrong')
    model.add_constraint(num_copy == dse.execution.dlrm.num_copy)
    
    model.add_constraint(TP == 1)
    model.add_constraint(PP == 1)
    model.add_constraint(DP == num_chips_per_copy)  

    Link_BW = model.add_matrix_variable(len(topology), name='Link_BW', vtype='continuous', lb=0)
    for i in range(len(topology)):
        model.add_constraint(Link_BW[i] == link_bw[i])      
            
    Link_BW_DP = model.add_variable(name='Link_BW_DP', vtype='continuous', lb=0)
    model.add_constraint(Link_BW_DP == highspy.min_(Link_BW[i] for i in range(len(topology))))
    
    
    
    
elif dse.execution.WhichOneof('workload_variant') == 'llm':



    is_TP_hierarchical = False
    is_DP_hierarchical = False



    ALL_REDUCE_ratio = model.add_variable(name='ALL_REDUCE_ratio', vtype='continuous', lb=0)
    ALL_TO_ALL_ratio = model.add_variable(name='ALL_TO_ALL_ratio', vtype='continuous', lb=0)
    ALL_GATHER_ratio = model.add_variable(name='ALL_GATHER_ratio', vtype='continuous', lb=0)
    ALL_REDUCE_PERIODIC_ratio = model.add_variable(name='ALL_REDUCE_PERIODIC_ratio', vtype='continuous', lb=0)


    ALL_REDUCE_ratio_sp = model.add_variable(name='ALL_REDUCE_ratio_sp', vtype='continuous', lb=0)
    ALL_GATHER_ratio_sp = model.add_variable(name='ALL_GATHER_ratio_sp', vtype='continuous', lb=0)


    P2P_ratio = model.add_variable(name='P2P_ratio', vtype='continuous', lb=0)
    
    if len(topology) == 1: # 1D
        Shape = model.add_matrix_variable(1, name='Shape', vtype='integer', lb=0)
        model.add_constraint(Shape[0] == num_chip)
        
        Link_BW = model.add_matrix_variable(1, name='Link_BW', vtype='continuous', lb=0)
        for i in range(len(topology)):
            model.add_constraint(Link_BW[i] == link_bw[i])  
            
        Link_BW_TP = model.add_variable(name='Link_BW_TP', vtype='continuous', lb=0)
        Link_BW_PP = model.add_variable(name='Link_BW_PP', vtype='continuous', lb=0)
        Link_BW_DP = model.add_variable(name='Link_BW_DP', vtype='continuous', lb=0)    
        
        if par[0] == 'TP':
            model.add_constraint(TP == num_chip)
            model.add_constraint(PP == 1)
            model.add_constraint(DP == 1)   

            model.add_constraint(Link_BW_TP == Link_BW[0])
            
            aaa = model.add_variable(vtype='continuous')
            model.add_constraint(aaa == TP * Link_BW_TP)
            if topology[0] == BasicTopology.R.value:
                if dse.execution.separate_rs_ag_for_ar:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == (TP - 1)*2)
                else:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == TP - 1)
                model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP * 8 == TP * TP)
                model.add_constraint(ALL_GATHER_ratio * aaa == TP - 1)
            elif topology[0] == BasicTopology.FC.value:
                if dse.execution.separate_rs_ag_for_ar:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == 2)
                else:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == 1)
                model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP == 1)
                model.add_constraint(ALL_GATHER_ratio * aaa == 1)
            elif topology[0] == BasicTopology.SW.value:
                if dse.execution.separate_rs_ag_for_ar:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == (TP - 1)*2)
                else:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == TP - 1)
                model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP * 4 == TP)
                model.add_constraint(ALL_GATHER_ratio * aaa == TP - 1)
            else:
                raise Exception('Wrong!')
                
            model.add_constraint(ALL_REDUCE_PERIODIC_ratio == 0)
            model.add_constraint(P2P_ratio == 0)
            
        elif par[0] == 'PP':
            model.add_constraint(TP == 1)
            model.add_constraint(PP == num_chip)
            model.add_constraint(DP == 1)   

            model.add_constraint(Link_BW_PP == Link_BW[0])
            
            model.add_constraint(ALL_REDUCE_ratio == 0)
            model.add_constraint(ALL_TO_ALL_ratio == 0)
            model.add_constraint(ALL_GATHER_ratio == 0)
            model.add_constraint(ALL_REDUCE_PERIODIC_ratio == 0)
            model.add_constraint(P2P_ratio * Link_BW_PP == 1)
            
        elif par[0] == 'DP':
            model.add_constraint(TP == 1)
            model.add_constraint(PP == 1)
            model.add_constraint(DP == num_chip)   

            model.add_constraint(Link_BW_DP == Link_BW[0])
            
            model.add_constraint(ALL_REDUCE_ratio == 0)
            model.add_constraint(ALL_TO_ALL_ratio == 0)
            model.add_constraint(ALL_GATHER_ratio == 0)

            bbb = model.add_variable(vtype='continuous')
            model.add_constraint(bbb == DP * Link_BW_DP)
            if topology[0] == BasicTopology.R.value:
                if dse.execution.separate_rs_ag_for_ar:
                    model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == (DP - 1)*2)
                else:
                    model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == DP - 1)
            elif topology[0] == BasicTopology.FC.value:
                model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == 1)
            elif topology[0] == BasicTopology.SW.value:
                if dse.execution.separate_rs_ag_for_ar:
                    model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == (DP - 1)*2)
                else:
                    model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == DP - 1)
            else:
                raise Exception('Wrong!')
            
            model.add_constraint(P2P_ratio == 0)
            
        else:
            raise Exception('Wrong!')
            
    elif len(topology) == 2: # 2D
        Shape = model.add_matrix_variable(2, name='Shape', vtype='integer', lb=1)
        if dimension[0] == 0:
            pass
        else:
            model.add_constraint(Shape[0] == dimension[0])
        
        if dimension[1] == 0:
            pass
        else:
            model.add_constraint(Shape[1] == dimension[1])
            
        model.add_constraint(Shape[0] * Shape[1] == num_chip)
        
        Link_BW = model.add_matrix_variable(2, name='Link_BW', vtype='continuous', lb=0)
        for i in range(len(topology)):
            model.add_constraint(Link_BW[i] == link_bw[i])
            
        Link_BW_TP = model.add_variable(name='Link_BW_TP', vtype='continuous', lb=0)
        Link_BW_PP = model.add_variable(name='Link_BW_PP', vtype='continuous', lb=0)
        Link_BW_DP = model.add_variable(name='Link_BW_DP', vtype='continuous', lb=0)
        
        if par[0] == 'TP' and par[1] == 'PP':
            model.add_constraint(TP == Shape[0])
            model.add_constraint(PP == Shape[1])
            model.add_constraint(DP == 1)

            model.add_constraint(Link_BW_TP == Link_BW[0])
            model.add_constraint(Link_BW_PP == Link_BW[1])
            
            aaa = model.add_variable(vtype='continuous')
            model.add_constraint(aaa == TP * Link_BW_TP)
            if topology[0] == BasicTopology.R.value:
                if dse.execution.separate_rs_ag_for_ar:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == (TP - 1)*2)
                else:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == TP - 1)
                model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP * 8 == TP * TP)
                model.add_constraint(ALL_GATHER_ratio * aaa == TP - 1)
            elif topology[0] == BasicTopology.FC.value:
                if dse.execution.separate_rs_ag_for_ar:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == 2)
                else:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == 1)
                model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP == 1)
                model.add_constraint(ALL_GATHER_ratio * aaa == 1)
            elif topology[0] == BasicTopology.SW.value:
                if dse.execution.separate_rs_ag_for_ar:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == (TP - 1)*2)
                else:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == TP - 1)
                model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP * 4 == TP)
                model.add_constraint(ALL_GATHER_ratio * aaa == TP - 1)
            else:
                raise Exception('Wrong!')
            
            model.add_constraint(ALL_REDUCE_PERIODIC_ratio == 0)
            model.add_constraint(P2P_ratio * Link_BW_PP == 1)
            
        elif par[0] == 'TP' and par[1] == 'DP':
            model.add_constraint(TP == Shape[0])
            model.add_constraint(PP == 1)
            model.add_constraint(DP == Shape[1])

            model.add_constraint(Link_BW_TP == Link_BW[0])
            model.add_constraint(Link_BW_DP == Link_BW[1])

            aaa = model.add_variable(vtype='continuous')
            model.add_constraint(aaa == TP * Link_BW_TP)
            if topology[0] == BasicTopology.R.value:
                if dse.execution.separate_rs_ag_for_ar:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == (TP - 1)*2)
                else:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == TP - 1)
                model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP * 8 == TP * TP)
                model.add_constraint(ALL_GATHER_ratio * aaa == TP - 1)
            elif topology[0] == BasicTopology.FC.value:
                if dse.execution.separate_rs_ag_for_ar:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == 2)
                else:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == 1)
                model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP == 1)
                model.add_constraint(ALL_GATHER_ratio * aaa == 1)
            elif topology[0] == BasicTopology.SW.value:
                if dse.execution.separate_rs_ag_for_ar:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == (TP - 1)*2)
                else:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == TP - 1)
                model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP * 4 == TP)
                model.add_constraint(ALL_GATHER_ratio * aaa == TP - 1)
            else:
                raise Exception('Wrong!')
                
            bbb = model.add_variable(vtype='continuous')
            model.add_constraint(bbb == DP * Link_BW_DP)
            if topology[1] == BasicTopology.R.value:
                if dse.execution.separate_rs_ag_for_ar:
                    model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == (DP - 1)*2)
                else:
                    model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == DP - 1)
            elif topology[1] == BasicTopology.FC.value:
                model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == 1)
            elif topology[1] == BasicTopology.SW.value:
                if dse.execution.separate_rs_ag_for_ar:
                    model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == (DP - 1)*2)
                else:
                    model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == DP - 1)
            else:
                raise Exception('Wrong!')
            
            model.add_constraint(P2P_ratio == 0)
                
        elif par[0] == 'PP' and par[1] == 'DP':
            model.add_constraint(TP == 1)
            model.add_constraint(PP == Shape[0])
            model.add_constraint(DP == Shape[1])

            model.add_constraint(Link_BW_PP == Link_BW[0])
            model.add_constraint(Link_BW_DP == Link_BW[1])
            
            model.add_constraint(ALL_REDUCE_ratio == 0)
            model.add_constraint(ALL_TO_ALL_ratio == 0)
            model.add_constraint(ALL_GATHER_ratio == 0)

            bbb = model.add_variable(vtype='continuous')
            model.add_constraint(bbb == DP * Link_BW_DP)
            if topology[1] == BasicTopology.R.value:
                if dse.execution.separate_rs_ag_for_ar:
                    model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == (DP - 1)*2)
                else:
                    model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == DP - 1)
            elif topology[1] == BasicTopology.FC.value:
                model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == 1)
            elif topology[1] == BasicTopology.SW.value:
                if dse.execution.separate_rs_ag_for_ar:
                    model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == (DP - 1)*2)
                else:
                    model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == DP - 1)
            else:
                raise Exception('Wrong!')
                
            model.add_constraint(P2P_ratio * Link_BW_PP == 1)
            
        else:    
            raise Exception('Wrong!')
            
    elif len(topology) == 3: # 3D
        Shape = model.add_matrix_variable(3, name='Shape', vtype='integer', lb=1)
        if dimension[0] == 0:
            pass
        else:
            model.add_constraint(Shape[0] == dimension[0])
        
        if dimension[1] == 0:
            pass
        else:
            model.add_constraint(Shape[1] == dimension[1])
        
        if dimension[2] == 0:
            pass
        else:
            model.add_constraint(Shape[2] == dimension[2])
            
        aaa = model.add_variable(vtype='continuous')
        model.add_constraint(aaa == Shape[0] * Shape[1])
        model.add_constraint(aaa * Shape[2] == num_chip)
        
        
        Link_BW = model.add_matrix_variable(3, name='Link_BW', vtype='continuous', lb=0)
        for i in range(len(topology)):
            model.add_constraint(Link_BW[i] == link_bw[i])  

        Link_BW_TP = model.add_variable(name='Link_BW_TP', vtype='continuous', lb=0)
        Link_BW_PP = model.add_variable(name='Link_BW_PP', vtype='continuous', lb=0)
        Link_BW_DP = model.add_variable(name='Link_BW_DP', vtype='continuous', lb=0)
        
        if par[0] == 'TP' and par[1] == 'PP' and par[2] == 'DP':
            model.add_constraint(TP == Shape[0])
            model.add_constraint(PP == Shape[1])
            model.add_constraint(DP == Shape[2])
            
            model.add_constraint(Link_BW_TP == Link_BW[0])
            model.add_constraint(Link_BW_PP == Link_BW[1])
            model.add_constraint(Link_BW_DP == Link_BW[2])
            
            

            aaa = model.add_variable(vtype='continuous')
            model.add_constraint(aaa == TP * Link_BW_TP)
            if topology[0] == BasicTopology.R.value:
                model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP * 8 == TP * TP)
                model.add_constraint(ALL_GATHER_ratio * aaa == TP - 1)
            elif topology[0] == BasicTopology.FC.value:
                model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP == 1)
                model.add_constraint(ALL_GATHER_ratio * aaa == 1)
            elif topology[0] == BasicTopology.SW.value:
                model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP * 4 == TP)
                model.add_constraint(ALL_GATHER_ratio * aaa == TP - 1)
            else:
                raise Exception('Wrong!')
            


            if dse.system.HBD == 0: # not set
                aaa = model.add_variable(vtype='continuous')
                model.add_constraint(aaa == TP * Link_BW_TP)
                if topology[0] == BasicTopology.R.value:
                    if dse.execution.separate_rs_ag_for_ar:
                        model.add_constraint(ALL_REDUCE_ratio * aaa == (TP - 1)*2)
                    else:
                        model.add_constraint(ALL_REDUCE_ratio * aaa == TP - 1)
                elif topology[0] == BasicTopology.FC.value:
                    if dse.execution.separate_rs_ag_for_ar:
                        model.add_constraint(ALL_REDUCE_ratio * aaa == 2)
                    else:
                        model.add_constraint(ALL_REDUCE_ratio * aaa == 1)
                elif topology[0] == BasicTopology.SW.value:
                    if dse.execution.separate_rs_ag_for_ar:
                        model.add_constraint(ALL_REDUCE_ratio * aaa == (TP - 1)*2)
                    else:
                        model.add_constraint(ALL_REDUCE_ratio * aaa == TP - 1)
                else:
                    raise Exception('Wrong!')
                               
                bbb = model.add_variable(vtype='continuous')
                model.add_constraint(bbb == DP * Link_BW_DP)
                if topology[2] == BasicTopology.R.value:
                    if dse.execution.separate_rs_ag_for_ar:
                        model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == (DP - 1)*2)
                    else:
                        model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == DP - 1)
                elif topology[2] == BasicTopology.FC.value:
                    model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == 1)
                elif topology[2] == BasicTopology.SW.value:
                    if dse.execution.separate_rs_ag_for_ar:
                        model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == (DP - 1)*2)
                    else:
                        model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == DP - 1)
                else:
                    raise Exception('Wrong!')
                
            else:
                # for TP
                if dse.system.HBD >= dimension[0]:
                    model.add_constraint(ALL_REDUCE_ratio * dimension[0] * Link_BW[0] == dimension[0] - 1)

                else:
                    # fast domain: HBD (x), Link_BW[0]
                    # slow domain: dimension[0]/HBD (y), Link_BW[1]
                    is_TP_hierarchical = True
                    
                    ALL_REDUCE_ratio_fast = model.add_variable(name='ALL_REDUCE_ratio_fast', vtype='continuous', lb=0)
                    ALL_REDUCE_ratio_slow = model.add_variable(name='ALL_REDUCE_ratio_slow', vtype='continuous', lb=0)
                    
                    model.add_constraint(ALL_REDUCE_ratio_fast * dse.system.HBD * Link_BW[0] >= dse.system.HBD - 1)
                    model.add_constraint(ALL_REDUCE_ratio_slow * dimension[0] * Link_BW[1] >= dimension[0]/dse.system.HBD - 1)
                
                
                # for DP
                if dse.system.HBD >= dimension[2]:
                    model.add_constraint(ALL_REDUCE_PERIODIC_ratio * dimension[2] * Link_BW[0] == dimension[2] - 1)
                    
                else:
                    # fast domain: HBD (x), Link_BW[0]
                    # slow domain: dimension[2]/HBD (y), Link_BW[2]
                    is_DP_hierarchical = True
                    
                    ALL_REDUCE_PERIODIC_ratio_fast = model.add_variable(name='ALL_REDUCE_PERIODIC_ratio_fast', vtype='continuous', lb=0)
                    ALL_REDUCE_PERIODIC_ratio_slow = model.add_variable(name='ALL_REDUCE_PERIODIC_ratio_slow', vtype='continuous', lb=0)
                    
                    model.add_constraint(ALL_REDUCE_PERIODIC_ratio_fast * dse.system.HBD * Link_BW[0] >= dse.system.HBD - 1)
                    model.add_constraint(ALL_REDUCE_PERIODIC_ratio_slow * dimension[2] * Link_BW[2] >= dimension[2]/dse.system.HBD - 1)
                
                
                
                
            
            model.add_constraint(P2P_ratio * Link_BW_PP == 1)
            
        else:
            raise Exception('Wrong!')
    
    elif len(topology) == 4: # 4D




        Shape = model.add_matrix_variable(4, name='Shape', vtype='integer', lb=1)
        if dimension[0] == 0:
            pass
        else:
            model.add_constraint(Shape[0] == dimension[0])
        
        if dimension[1] == 0:
            pass
        else:
            model.add_constraint(Shape[1] == dimension[1])
        
        if dimension[2] == 0:
            pass
        else:
            model.add_constraint(Shape[2] == dimension[2])

        if dimension[3] == 0:
            pass
        else:
            model.add_constraint(Shape[3] == dimension[3])
            
        aaa = model.add_variable(vtype='continuous')
        bbb = model.add_variable(vtype='continuous')
        model.add_constraint(aaa == Shape[0] * Shape[1])
        model.add_constraint(bbb == Shape[2] * Shape[3])
        model.add_constraint(aaa * bbb == num_chip)
        
        
        Link_BW = model.add_matrix_variable(4, name='Link_BW', vtype='continuous', lb=0)
        for i in range(len(topology)):
            model.add_constraint(Link_BW[i] == link_bw[i])  

        Link_BW_SP = model.add_variable(name='Link_BW_SP', vtype='continuous', lb=0)
        Link_BW_TP = model.add_variable(name='Link_BW_TP', vtype='continuous', lb=0)
        Link_BW_PP = model.add_variable(name='Link_BW_PP', vtype='continuous', lb=0)
        Link_BW_DP = model.add_variable(name='Link_BW_DP', vtype='continuous', lb=0)
        
        if par[0] == 'SP' and par[1] == 'TP' and par[2] == 'PP' and par[3] == 'DP':
            model.add_constraint(SP == Shape[0])
            model.add_constraint(TP == Shape[1])
            model.add_constraint(PP == Shape[2])
            model.add_constraint(DP == Shape[3])
            
            model.add_constraint(Link_BW_SP == Link_BW[0])
            model.add_constraint(Link_BW_TP == Link_BW[1])
            model.add_constraint(Link_BW_PP == Link_BW[2])
            model.add_constraint(Link_BW_DP == Link_BW[3])
            
            

            aaa = model.add_variable(vtype='continuous')
            model.add_constraint(aaa == TP * Link_BW_TP)
            if topology[1] == BasicTopology.R.value:
                model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP * 8 == TP * TP)
                model.add_constraint(ALL_GATHER_ratio * aaa == TP - 1)
            elif topology[1] == BasicTopology.FC.value:
                model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP == 1)
                model.add_constraint(ALL_GATHER_ratio * aaa == 1)
            elif topology[1] == BasicTopology.SW.value:
                model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP * 4 == TP)
                model.add_constraint(ALL_GATHER_ratio * aaa == TP - 1)
            else:
                raise Exception('Wrong!')
            


            aaa = model.add_variable(vtype='continuous')
            model.add_constraint(aaa == TP * Link_BW_TP)
            if topology[1] == BasicTopology.R.value:
                if dse.execution.separate_rs_ag_for_ar:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == (TP - 1)*2)
                else:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == TP - 1)
            elif topology[1] == BasicTopology.FC.value:
                if dse.execution.separate_rs_ag_for_ar:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == 2)
                else:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == 1)
            elif topology[1] == BasicTopology.SW.value:
                if dse.execution.separate_rs_ag_for_ar:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == (TP - 1)*2)
                else:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == TP - 1)
            else:
                raise Exception('Wrong!')
                            
            bbb = model.add_variable(vtype='continuous')
            model.add_constraint(bbb == DP * Link_BW_DP)
            if topology[3] == BasicTopology.R.value:
                if dse.execution.separate_rs_ag_for_ar:
                    model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == (DP - 1)*2)
                else:
                    model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == DP - 1)
            elif topology[3] == BasicTopology.FC.value:
                model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == 1)
            elif topology[3] == BasicTopology.SW.value:
                if dse.execution.separate_rs_ag_for_ar:
                    model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == (DP - 1)*2)
                else:
                    model.add_constraint(ALL_REDUCE_PERIODIC_ratio * bbb == DP - 1)
            else:
                raise Exception('Wrong!')
            











            aaa = model.add_variable(vtype='continuous')
            model.add_constraint(aaa == SP * Link_BW_SP)
            if topology[0] == BasicTopology.R.value:
                model.add_constraint(ALL_GATHER_ratio_sp * aaa == SP - 1)
            elif topology[0] == BasicTopology.FC.value:
                model.add_constraint(ALL_GATHER_ratio_sp * aaa == 1)
            elif topology[0] == BasicTopology.SW.value:
                model.add_constraint(ALL_GATHER_ratio_sp * aaa == SP - 1)
            else:
                raise Exception('Wrong!')
            


            aaa = model.add_variable(vtype='continuous')
            model.add_constraint(aaa == SP * Link_BW_SP)
            if topology[0] == BasicTopology.R.value:
                if dse.execution.separate_rs_ag_for_ar:
                    model.add_constraint(ALL_REDUCE_ratio_sp * aaa == (SP - 1)*2)
                else:
                    model.add_constraint(ALL_REDUCE_ratio_sp * aaa == SP - 1)
            elif topology[0] == BasicTopology.FC.value:
                if dse.execution.separate_rs_ag_for_ar:
                    model.add_constraint(ALL_REDUCE_ratio_sp * aaa == 2)
                else:
                    model.add_constraint(ALL_REDUCE_ratio_sp * aaa == 1)
            elif topology[0] == BasicTopology.SW.value:
                if dse.execution.separate_rs_ag_for_ar:
                    model.add_constraint(ALL_REDUCE_ratio_sp * aaa == (SP - 1)*2)
                else:
                    model.add_constraint(ALL_REDUCE_ratio_sp * aaa == SP - 1)
            else:
                raise Exception('Wrong!')


                
            model.add_constraint(P2P_ratio * Link_BW_PP == 1)

        else:
            raise Exception('Wrong!')







        
    else:
        raise Exception('Wrong!')
    
    
    model.add_constraint(num_copy == 1)
    
    
    
    


layer_per_stage = model.add_variable(name='layer_per_stage', vtype='integer', lb=1)
layers = model.add_variable(name='layers', vtype='integer', lb=1)

if dse.execution.WhichOneof('workload_variant') == 'llm':
    if dse.execution.llm.num_layer_in_graph == 0:
        raise Exception("Wrong!")
    elif dse.execution.llm.num_layer_in_graph == 1:
        tmp = int(dse.execution.llm.num_layer)
        model.add_constraint(layers == tmp)
    else:
        model.add_constraint(layers == PP)
model.add_constraint(layer_per_stage * PP >= layers)









# tiling
tile_size = model.add_variable(name='tile_size', vtype='integer', lb=1)
num_tile = model.add_variable(name='num_tile', vtype='integer', lb=1)

if dse.execution.WhichOneof('workload_variant') == 'llm':    
    if dse.execution.llm.tile_size == 0:
        pass
    else:
        model.add_constraint(tile_size == dse.execution.llm.tile_size)

    if len(dimension) == 4:
        sp_dim = dimension[0]
        seq_len = seq_len // sp_dim
    
    model.add_constraint(tile_size * num_tile >= seq_len)
    
elif dse.execution.WhichOneof('workload_variant') == 'dlrm' or dse.execution.WhichOneof('workload_variant') == 'hpl' or dse.execution.WhichOneof('workload_variant') == 'fft':
    model.add_constraint(num_tile == 1)
    
else:
    raise Exception('Wrong!')




# sharding kernels
shard_M = model.add_matrix_variable(num_node, name='shard_M', vtype='integer', lb=0)
shard_K = model.add_matrix_variable(num_node, name='shard_K', vtype='integer', lb=0)
shard_N = model.add_matrix_variable(num_node, name='shard_N', vtype='integer', lb=0)

if dse.execution.WhichOneof('workload_variant') == 'hpl':
    for i in range(num_node):
        model.add_constraint(shard_M[i] * HPL_sharding_factor_X >= M[i])
        model.add_constraint(shard_K[i] * HPL_sharding_factor_Y >= K[i])
        model.add_constraint(shard_N[i] == N[i])
 
elif dse.execution.WhichOneof('workload_variant') == 'fft':
    model.add_constraint(shard_M[0] == M[0])
    model.add_constraint(shard_K[0] == K[0])
    model.add_constraint(shard_N[0] * num_chips_per_copy >= N[0])  
     
    model.add_constraint(shard_M[1] == M[1])
    model.add_constraint(shard_K[1] == K[1])
    model.add_constraint(shard_N[1] * num_chips_per_copy >= N[1])
    
else:
    for i in range(num_node):
        if sharding[i] == Dim.OUTER_DIM.value or sharding[i] == Dim.M_DIM.value:
            if tiling[i] == Dim.OUTER_DIM.value or tiling[i] == Dim.M_DIM.value:
                raise Exception('Wrong!')
            
            elif tiling[i] == Dim.K_DIM.value:
                model.add_constraint(shard_M[i] * TP >= M[i])
                model.add_constraint(shard_K[i] * num_tile >= K[i])
                model.add_constraint(shard_N[i] >= N[i])
                
            elif tiling[i] == Dim.N_DIM.value:
                model.add_constraint(shard_M[i] * TP >= M[i])
                model.add_constraint(shard_K[i] >= K[i])
                model.add_constraint(shard_N[i] * num_tile >= N[i])
            
            elif tiling[i] == Dim.NO_DIM.value:    
                model.add_constraint(shard_M[i] * TP >= M[i])
                model.add_constraint(shard_K[i] >= K[i])
                model.add_constraint(shard_N[i] >= N[i])
                
            else:
                raise Exception('Wrong!')
     
        elif sharding[i] == Dim.K_DIM.value:
            if tiling[i] == Dim.OUTER_DIM.value or tiling[i] == Dim.M_DIM.value:
                model.add_constraint(shard_M[i] * num_tile >= M[i])
                model.add_constraint(shard_K[i] * TP >= K[i])
                model.add_constraint(shard_N[i] >= N[i])
            
            elif tiling[i] == Dim.K_DIM.value:
                raise Exception('Wrong!')
                
            elif tiling[i] == Dim.N_DIM.value:
                model.add_constraint(shard_M[i] >= M[i])
                model.add_constraint(shard_K[i] * TP >= K[i])
                model.add_constraint(shard_N[i] * num_tile >= N[i])
                
            elif tiling[i] == Dim.NO_DIM.value:    
                model.add_constraint(shard_M[i] >= M[i])
                model.add_constraint(shard_K[i] * TP >= K[i])
                model.add_constraint(shard_N[i] >= N[i])
            
            else:
                raise Exception('Wrong!')
            
        elif sharding[i] == Dim.N_DIM.value:
            if tiling[i] == Dim.OUTER_DIM.value or tiling[i] == Dim.M_DIM.value:
                model.add_constraint(shard_M[i] * num_tile >= M[i])
                model.add_constraint(shard_K[i] >= K[i])
                model.add_constraint(shard_N[i] * TP >= N[i])
            
            elif tiling[i] == Dim.K_DIM.value:
                model.add_constraint(shard_M[i] >= M[i])
                model.add_constraint(shard_K[i] * num_tile >= K[i])
                model.add_constraint(shard_N[i] * TP >= N[i])
                
            elif tiling[i] == Dim.N_DIM.value:
                raise Exception('Wrong!')
                
            elif tiling[i] == Dim.NO_DIM.value:    
                model.add_constraint(shard_M[i] >= M[i])
                model.add_constraint(shard_K[i] >= K[i])
                model.add_constraint(shard_N[i] * TP >= N[i])
                
            else:
                raise Exception('Wrong!')
            
        elif sharding[i] == Dim.NO_DIM.value:
            if tiling[i] == Dim.OUTER_DIM.value or tiling[i] == Dim.M_DIM.value:
                model.add_constraint(shard_M[i] * num_tile >= M[i])
                model.add_constraint(shard_K[i] >= K[i])
                model.add_constraint(shard_N[i] >= N[i])
            
            elif tiling[i] == Dim.K_DIM.value:
                model.add_constraint(shard_M[i] >= M[i])
                model.add_constraint(shard_K[i] * num_tile >= K[i])
                model.add_constraint(shard_N[i] >= N[i])
                
            elif tiling[i] == Dim.N_DIM.value:
                model.add_constraint(shard_M[i] >= M[i])
                model.add_constraint(shard_K[i] >= K[i])
                model.add_constraint(shard_N[i] * num_tile >= N[i])
            
            elif tiling[i] == Dim.NO_DIM.value:    
                model.add_constraint(shard_M[i] >= M[i])
                model.add_constraint(shard_K[i] >= K[i])
                model.add_constraint(shard_N[i] >= N[i])
            
            else:
                raise Exception('Wrong!')
            
        else:
            raise Exception('Wrong!')



# sharding intermediate buffers
shard_intermediate_buffer_size = model.add_matrix_variable(num_edge, name='shard_intermediate_buffer_size', vtype='continuous', lb=0)
for i in range(num_edge):
    upstream_node_idx = node_dict[startIdx[i]]
    if zero_out[i]:
        model.add_constraint(shard_intermediate_buffer_size[i] == 0)
    else:
        model.add_constraint(shard_intermediate_buffer_size[i] == shard_M[upstream_node_idx] * shard_N[upstream_node_idx] * word)

# sharding initiation buffers (weights)
shard_initiation_buffer_size = model.add_matrix_variable(num_weight, name='shard_initiation_buffer_size', vtype='continuous', lb=0)
for i in range(num_weight):
    node_idx = node_dict[weight_dict[i]]
    model.add_constraint(shard_initiation_buffer_size[i] == shard_M[node_idx] * shard_K[node_idx] * word)




Micro_Batch_Size = model.add_variable(name='Micro_Batch_Size', vtype='integer', lb=1)

if micro_batch_size == 0:
    pass
else:
    model.add_constraint(Micro_Batch_Size == micro_batch_size)
    
num_micro_batch_per_pipeline = model.add_variable(name='num_micro_batch_per_pipeline', vtype='integer', lb=0)
aaa = model.add_variable(vtype='integer')
model.add_constraint(aaa == Micro_Batch_Size * DP)
model.add_constraint(num_micro_batch_per_pipeline * aaa == global_batch_size)





# sharding communication
if dse.execution.WhichOneof('workload_variant') == 'llm':
    ALL_REDUCE_communication_size_node = model.add_matrix_variable(num_node, name='ALL_REDUCE_communication_size_node', vtype='continuous', lb=0)
    ALL_TO_ALL_communication_size_node = model.add_matrix_variable(num_node, name='ALL_TO_ALL_communication_size_node', vtype='continuous', lb=0)
    ALL_GATHER_communication_size_node = model.add_matrix_variable(num_node, name='ALL_GATHER_communication_size_node', vtype='continuous', lb=0)
    ALL_REDUCE_PERIODIC_communication_size_node = model.add_matrix_variable(num_node, name='ALL_REDUCE_PERIODIC_communication_size_node', vtype='continuous', lb=0)

    ALL_REDUCE_communication_size_node_sp = model.add_matrix_variable(num_node, name='ALL_REDUCE_communication_size_node_sp', vtype='continuous', lb=0)
    ALL_GATHER_communication_size_node_sp = model.add_matrix_variable(num_node, name='ALL_GATHER_communication_size_node_sp', vtype='continuous', lb=0)


    for i in range(num_node):
        if node_communication_type[i] == Communication.ALL_REDUCE.value:
            model.add_constraint(ALL_REDUCE_communication_size_node[i] == shard_M[i] * shard_N[i] * word)
        else:
            model.add_constraint(ALL_REDUCE_communication_size_node[i] == 0)
        
        if node_communication_type[i] == Communication.ALL_TO_ALL.value:
            model.add_constraint(ALL_TO_ALL_communication_size_node[i] == shard_M[i] * shard_N[i] * word)
        else:
            model.add_constraint(ALL_TO_ALL_communication_size_node[i] == 0)

        if node_communication_type[i] == Communication.ALL_GATHER.value:
            model.add_constraint(ALL_GATHER_communication_size_node[i] == shard_M[i] * shard_N[i] * word)
        else:
            model.add_constraint(ALL_GATHER_communication_size_node[i] == 0)

        if node_communication_type[i] == Communication.ALL_REDUCE_PERIODIC.value:
            aaa = model.add_variable(vtype='continuous')
            bbb = model.add_variable(vtype='continuous')
            ccc = model.add_variable(vtype='continuous')
            model.add_constraint(aaa == shard_M[i] * shard_N[i] * word)
            model.add_constraint(ALL_REDUCE_PERIODIC_communication_size_node[i] * num_micro_batch_per_pipeline >= aaa)
        else:
            model.add_constraint(ALL_REDUCE_PERIODIC_communication_size_node[i] == 0)

        



        if node_communication_type_2[i] == Communication.ALL_REDUCE.value:
            model.add_constraint(ALL_REDUCE_communication_size_node_sp[i] == shard_M[i] * shard_N[i] * word)
        else:
            model.add_constraint(ALL_REDUCE_communication_size_node_sp[i] == 0)

        if node_communication_type_2[i] == Communication.ALL_GATHER.value:
            model.add_constraint(ALL_GATHER_communication_size_node_sp[i] == shard_M[i] * shard_N[i] * word)
        else:
            model.add_constraint(ALL_GATHER_communication_size_node_sp[i] == 0)
    


    ALL_REDUCE_communication_size_edge = model.add_matrix_variable(num_edge, name='ALL_REDUCE_communication_size_edge', vtype='continuous', lb=0)
    ALL_TO_ALL_communication_size_edge = model.add_matrix_variable(num_edge, name='ALL_TO_ALL_communication_size_edge', vtype='continuous', lb=0)
    ALL_GATHER_communication_size_edge = model.add_matrix_variable(num_edge, name='ALL_GATHER_communication_size_edge', vtype='continuous', lb=0)
    
    ALL_REDUCE_communication_size_edge_sp = model.add_matrix_variable(num_edge, name='ALL_REDUCE_communication_size_edge_sp', vtype='continuous', lb=0)
    ALL_GATHER_communication_size_edge_sp = model.add_matrix_variable(num_edge, name='ALL_GATHER_communication_size_edge_sp', vtype='continuous', lb=0)
    
    for i in range(num_edge):
        start_node_idx = node_dict[startIdx[i]]
        
        if edge_communication_type[i] == Communication.ALL_REDUCE.value:
            model.add_constraint(ALL_REDUCE_communication_size_edge[i] == shard_M[start_node_idx] * shard_N[start_node_idx] * word)
        else:
            model.add_constraint(ALL_REDUCE_communication_size_edge[i] == 0)

        if edge_communication_type[i] == Communication.ALL_TO_ALL.value:
            model.add_constraint(ALL_TO_ALL_communication_size_edge[i] == shard_M[start_node_idx] * shard_N[start_node_idx] * word)
        else:
            model.add_constraint(ALL_TO_ALL_communication_size_edge[i] == 0)

        if edge_communication_type[i] == Communication.ALL_GATHER.value:
            model.add_constraint(ALL_GATHER_communication_size_edge[i] == shard_M[start_node_idx] * shard_N[start_node_idx] * word)
        else:
            model.add_constraint(ALL_GATHER_communication_size_edge[i] == 0)
        



        start_node_idx = node_dict[startIdx[i]]
        
        if edge_communication_type_2[i] == Communication.ALL_REDUCE.value:
            model.add_constraint(ALL_REDUCE_communication_size_edge_sp[i] == shard_M[start_node_idx] * shard_N[start_node_idx] * word)
        else:
            model.add_constraint(ALL_REDUCE_communication_size_edge_sp[i] == 0)

        if edge_communication_type_2[i] == Communication.ALL_GATHER.value:
            model.add_constraint(ALL_GATHER_communication_size_edge_sp[i] == shard_M[start_node_idx] * shard_N[start_node_idx] * word)
            print('aaaaaa', start_node_idx, kernel_name, kernel_name[start_node_idx])
        else:
            model.add_constraint(ALL_GATHER_communication_size_edge_sp[i] == 0)



elif dse.execution.WhichOneof('workload_variant') == 'dlrm':
    ALL_TO_ALL_communication_size_node = model.add_matrix_variable(num_node, name='ALL_TO_ALL_communication_size_node', vtype='continuous', lb=0)
    ALL_REDUCE_PERIODIC_communication_size_node = model.add_matrix_variable(num_node, name='ALL_REDUCE_PERIODIC_communication_size_node', vtype='continuous', lb=0)
    
    for i in range(num_node):
        if node_communication_type[i] == Communication.ALL_TO_ALL.value:
            aaa = model.add_variable(vtype='binary')
            tmp = model.add_variable(vtype='continuous')
            model.add_constraint((aaa == 1) >> (num_chips_per_copy == 1))
            model.add_constraint((aaa == 0) >> (num_chips_per_copy >= 2))
            model.add_constraint((aaa == 1) >> (ALL_TO_ALL_communication_size_node[i] == 0))
            model.add_constraint(tmp == ALL_TO_ALL_communication_size_node[i] * num_chips_per_copy)
            model.add_constraint((aaa == 0) >> (tmp >= node_communication_size[i]))
        else:
            model.add_constraint(ALL_TO_ALL_communication_size_node[i] == 0)
        
        if node_communication_type[i] == Communication.ALL_REDUCE_PERIODIC.value:
            aaa = model.add_variable(vtype='continuous')
            bbb = model.add_variable(vtype='continuous')
            ccc = model.add_variable(vtype='continuous')
            model.add_constraint(aaa == shard_M[i] * shard_N[i] * word)
            model.add_constraint(ALL_REDUCE_PERIODIC_communication_size_node[i] * num_micro_batch_per_pipeline >= aaa)
        else:
            model.add_constraint(ALL_REDUCE_PERIODIC_communication_size_node[i] == 0) 
    
elif dse.execution.WhichOneof('workload_variant') == 'hpl':
    BROADCAST_communication_size = model.add_matrix_variable(num_node, name='BROADCAST_communication_size', vtype='continuous', lb=0)
    POINT_TO_POINT_communication_size = model.add_matrix_variable(num_node, name='POINT_TO_POINT_communication_size', vtype='continuous', lb=0)
    
    for i in range(num_node):
        if node_communication_type[i] == Communication.BROADCAST.value:
            aaa = model.add_variable(vtype='binary')
            tmp = model.add_variable(vtype='continuous', lb=0)
            model.add_constraint((aaa == 1) >> (HPL_sharding_factor_X == 1))
            model.add_constraint((aaa == 0) >> (HPL_sharding_factor_X >= 2))
            model.add_constraint((aaa == 1) >> (BROADCAST_communication_size[i] == 0))
            model.add_constraint(tmp == BROADCAST_communication_size[i] * HPL_sharding_factor_X)
            model.add_constraint((aaa == 0) >> (tmp >= node_communication_size[i]))
        else:
            model.add_constraint(BROADCAST_communication_size[i] == 0)
            
        if node_communication_type_2[i] == Communication.POINT_TO_POINT.value:
            aaa = model.add_variable(vtype='binary')
            tmp = model.add_variable(vtype='continuous', lb=0)
            model.add_constraint((aaa == 1) >> (HPL_sharding_factor_Y == 1))
            model.add_constraint((aaa == 0) >> (HPL_sharding_factor_Y >= 2))
            model.add_constraint((aaa == 1) >> (POINT_TO_POINT_communication_size[i] == 0))
            model.add_constraint(tmp == POINT_TO_POINT_communication_size[i] * HPL_sharding_factor_Y)
            model.add_constraint((aaa == 0) >> (tmp >= node_communication_size[i]))
        else:
            model.add_constraint(POINT_TO_POINT_communication_size[i] == 0)

elif dse.execution.WhichOneof('workload_variant') == 'fft':
    ALL_TO_ALL_communication_size_node = model.add_matrix_variable(num_node, name='ALL_TO_ALL_communication_size_node', vtype='continuous', lb=0)
    
    for i in range(num_node):
        if node_communication_type[i] == Communication.ALL_TO_ALL.value:
            aaa = model.add_variable(vtype='binary')
            tmp = model.add_variable(vtype='continuous', lb=0)
            tmp2 = model.add_variable(vtype='continuous', lb=0)
            model.add_constraint(tmp == num_chips_per_copy * num_chips_per_copy)
            model.add_constraint(tmp2 == ALL_TO_ALL_communication_size_node[i] * tmp)
            model.add_constraint((aaa == 1) >> (num_chips_per_copy == 1))
            model.add_constraint((aaa == 0) >> (num_chips_per_copy >= 2))
            model.add_constraint((aaa == 1) >> (ALL_TO_ALL_communication_size_node[i] == 0))
            model.add_constraint((aaa == 0) >> (tmp2 >= node_communication_size[i]))
        else:
            model.add_constraint(ALL_TO_ALL_communication_size_node[i] == 0)
              
else:
    raise Exception('Wrong!')





















if dse.execution.num_config == 0: # not specified
    C = num_node
else:
    C = dse.execution.num_config

    
Config = model.add_matrix_variable(num_node, name='Config', vtype='integer', lb=0)

A = model.add_matrix_variable((num_node, C), name='A', vtype='binary') # kernel to config
B = model.add_matrix_variable((num_edge, C), name='B', vtype='binary') # on-chip buffer to config
D = model.add_matrix_variable((num_edge, C), name='D', vtype='binary') # off-chip buffer to config

Z = model.add_matrix_variable((num_edge, C), name='Z', vtype='binary') # on/off-chip buffer to config

E = model.add_matrix_variable((num_edge, C), name='E', vtype='binary') # DRAM buffer to config
H = model.add_matrix_variable((num_edge, C), name='H', vtype='binary') # buffer placement based on upstream kernel placement

F = model.add_matrix_variable((num_weight, C), name='F', vtype='binary') # initiation buffer to config

if dse.execution.execution_style == Execution_Style.KERNEL_BY_KERNEL.value:
    for i in range(len(configs)):
        model.add_constraint(Config[i] == i)
    
elif dse.execution.execution_style == Execution_Style.DATAFLOW.value:
    if C == num_node:
        for i in range(len(configs)):
            model.add_constraint(Config[i] == i)

    else:
        for i in range(len(configs)):
            if configs[i] != -1:
                model.add_constraint(Config[i] == configs[i])
        
        if first_bwd_kernel != -1: # if config is not specified for dataflow or there is backward pass for training
            if C % 2 == 0:
                fwd_idx = int(C / 2) - 1
                bwd_idx = int(C / 2)
            elif C % 2 == 1:
                fwd_idx = math.floor(C / 2)
                bwd_idx = math.floor(C / 2)
            else:
                raise Exception('Wrong!')
            
            for i in range(num_node):
                if fwd_bwd[i] == FWD_BWD.FWD.value:
                    model.add_constraint(Config[i] <= fwd_idx)
                elif fwd_bwd[i] == FWD_BWD.BWD.value:
                    model.add_constraint(Config[i] >= bwd_idx)
                else:
                    raise Exception('Wrong!')
        
else:
    raise Exception('Wrong!')





# kernel assignment   
for i in range(num_node):
    model.add_constraint(A[i, :] @ np.ones((C)) == 1)
    
    
t2 = np.zeros((C))
for i in range(C):
    t2[i] = i
for i in range(num_node):
    model.add_constraint(A[i, :] @ t2 == Config[i])


for i in range(num_edge):
    model.add_constraint(Config[node_dict[startIdx[i]]] <= Config[node_dict[endIdx[i]]])


            
 
if dse.execution.execution_style == Execution_Style.KERNEL_BY_KERNEL.value:
    for i in range(C):
        model.add_constraint(np.ones((num_node)) @ A[:, i] >= 1)
    model.add_constraint(C == num_node)
        
elif dse.execution.execution_style == Execution_Style.DATAFLOW.value:
    pass
    
else:
    raise Exception('Wrong!')







flag = False
for i in range(len(num_input_set)):
    if num_input_set[i]:
        flag = True

for i in range(len(num_input_set)):
    for j in range(i, len(num_input_set)):
        if (num_input_set[i] == False and num_input_set[j] == True) or (num_input_set[i] == True and num_input_set[j] == False):
            b = model.add_variable(vtype='binary')
            model.add_constraint(Config[i] <= Config[j] - 1 + large_number * (1 - b))
            model.add_constraint(Config[i] >= Config[j] + 1 - large_number * b)





num_input_per_config = model.add_matrix_variable(C, name='num_input_per_config', vtype='integer')
for i in range(C):
    aaa = model.add_matrix_variable(num_node, vtype='continuous')
    for j in range(num_node):
        model.add_constraint(aaa[j] == num_input[j] * A[j, i])
    model.add_constraint(num_input_per_config[i] == highspy.max_(aaa[k] for k in range(num_node)))



num_tile_per_config = model.add_matrix_variable(C, name='num_tile_per_config', vtype='integer')
if flag:
    tmp = np.zeros((num_node))
    for i in range(len(num_input_set)):
        if num_input_set[i]:
            tmp[i] = 1
        else:
            tmp[i] = 0

    zzzz = model.add_matrix_variable(C, name='zzzz', vtype='integer')
    for i in range(C):
        model.add_constraint(zzzz[i] == tmp @ A[:, i])
                        
        aaa = model.add_variable(vtype='binary')
        model.add_constraint((aaa == 1) >> (zzzz[i] >= 1))
        model.add_constraint((aaa == 0) >> (zzzz[i] <= 0))
        
        model.add_constraint((aaa == 1) >> (num_tile_per_config[i] == 1))
        model.add_constraint((aaa == 0) >> (num_tile_per_config[i] == num_tile))             

else:
    for i in range(C):
        model.add_constraint(num_tile_per_config[i] == num_tile)

# compute resources
if dse.execution.compute_util == 0:
    Par_lane = model.add_matrix_variable((num_node), name='Par_lane', vtype='integer', lb=1)
    Par_stage = model.add_matrix_variable((num_node), name='Par_stage', vtype='integer', lb=1)
else:
    pass

Par_total = model.add_matrix_variable((num_node), name='Par_total', vtype='integer', lb=1)

if dse.execution.compute_util == 0:
    for i in range(num_node):
        model.add_constraint(Par_lane[i] * Par_stage[i] == Par_total[i])
else:
    pass
    
for i in range(C):
    model.add_constraint(Par_total @ A[:, i] <= Core)   




# intermediate buffer assignment
for i in range(num_edge):
    start_node_idx = node_dict[startIdx[i]]
    end_node_idx = node_dict[endIdx[i]]
    
    for j in range(C):
        t1 = model.add_variable(vtype='binary')
        t2 = model.add_variable(vtype='binary')
        t3 = model.add_variable(vtype='binary')
        t4 = model.add_variable(vtype='binary')
        t5 = model.add_variable(vtype='binary')

        model.add_constraint(t1 == highspy.and_(A[start_node_idx, j], A[end_node_idx, j]))
        model.add_constraint(t2 == highspy.or_(A[start_node_idx, j], A[end_node_idx, j]))
        model.add_constraint(t3 == 1 - t1)
        model.add_constraint(t4 == highspy.and_(t3, t2))
        model.add_constraint(t5 == A[start_node_idx, j])

        model.add_constraint((t1 == 1) >> (B[i, j] == 1))
        model.add_constraint((t1 == 0) >> (B[i, j] == 0))
        
        model.add_constraint((t4 == 1) >> (D[i, j] == 1))
        model.add_constraint((t4 == 0) >> (D[i, j] == 0))
        
        model.add_constraint((t2 == 1) >> (Z[i, j] == 1))
        model.add_constraint((t2 == 0) >> (Z[i, j] == 0))

        model.add_constraint((t5 == 1) >> (H[i, j] == 1))
        model.add_constraint((t5 == 0) >> (H[i, j] == 0))



# ls, lt, utilities for E matrix
ls = np.zeros(shape=(C, C))
lt = np.zeros(shape=(C, C))

for i in range(C):
    for j in range(C):
        if i <= j:
            ls[i, j] = 1
        if i < j:
            lt[i, j] = 1


for i in range(num_edge):
    start_node_idx = node_dict[startIdx[i]]
    end_node_idx = node_dict[endIdx[i]]
    
    tmp1 = model.add_matrix_variable((C), vtype='binary')
    tmp2 = model.add_matrix_variable((C), vtype='binary')

    model.add_constraint(tmp1 == A[start_node_idx, :] @ ls)
    model.add_constraint(tmp2 == A[end_node_idx, :] @ lt)
    
    for j in range(C):
        t1 = model.add_variable(vtype='binary')
        t2 = model.add_variable(vtype='binary')
        t3 = model.add_variable(vtype='binary')
        t4 = model.add_variable(vtype='binary')
        t5 = model.add_variable(vtype='binary')
        t6 = model.add_variable(vtype='binary')
        t7 = model.add_variable(vtype='binary')
        t8 = model.add_variable(vtype='binary')
        t9 = model.add_variable(vtype='binary')

        model.add_constraint(t1 == highspy.and_(tmp1[j], tmp2[j]))
        model.add_constraint(t2 == highspy.or_(tmp1[j], tmp2[j]))
        model.add_constraint(t3 == 1 - t1)
        model.add_constraint(t4 == highspy.and_(t2, t3))
        
        model.add_constraint(t5 == highspy.and_(A[start_node_idx, j], A[end_node_idx, j]))
        
        model.add_constraint(t6 == highspy.and_(t4, t5))
        model.add_constraint(t7 == highspy.or_(t4, t5))
        model.add_constraint(t8 == 1 - t6)
        model.add_constraint(t9 == highspy.and_(t7, t8))
        
        
        model.add_constraint((t9 == 1) >> (E[i, j] == 1))
        model.add_constraint((t9 == 0) >> (E[i, j] == 0))
        
        
        
# initiation buffer assignment
for i in range(num_weight):
    node_idx = node_dict[weight_dict[i]]
    
    for j in range(C):
        model.add_constraint((A[node_idx, j] == 1) >> (F[i, j] == 1))
        model.add_constraint((A[node_idx, j] == 0) >> (F[i, j] == 0))









if dse.system.accelerator.sram_cap == 0:
    sram_cap = sys.maxsize
else:
    sram_cap = dse.system.accelerator.sram_cap



if dse.system.accelerator.pmu == 0:
    p_and_r_flag = False
    
    # SRAM cap
    shard_intermediate_buffer_size_depth_original = model.add_matrix_variable(num_edge, name='shard_intermediate_buffer_size_depth_original', vtype='integer', lb=0)
    shard_intermediate_buffer_size_depth_two = model.add_matrix_variable(num_edge, name='shard_intermediate_buffer_size_depth_two', vtype='integer', lb=0)
    shard_initiation_buffer_size_depth_one = model.add_matrix_variable(num_weight, name='shard_initiation_buffer_size_depth_one', vtype='integer', lb=0)


    # output edge dict: map node idx to output edge idx
    downstream_edge_dict = dict()
    for i in range(num_edge):
        start_node_idx = node_dict[startIdx[i]]
        end_node_idx = node_dict[endIdx[i]]
        
        if start_node_idx not in downstream_edge_dict.keys():
            downstream_edge_dict[start_node_idx] = [i]
        else:
            downstream_edge_dict[start_node_idx].append(i)



    tiled_kernel_list = []
    tiled_edge_list = []
    tiling_factor = model.add_matrix_variable(num_node, name='tiling_factor', vtype='integer', lb=1)  
    for i in range(num_weight):
        node_idx = node_dict[weight_dict[i]]
        tiled_kernel_list.append(node_idx)
        
        if skip_weight[node_idx]:
            model.add_constraint(shard_initiation_buffer_size_depth_one[i] == 0)
        else:
            model.add_constraint(shard_initiation_buffer_size_depth_one[i] * tiling_factor[node_idx] == shard_initiation_buffer_size[i] * 1)
        
        if node_idx in downstream_edge_dict.keys():
            for downstream_tensor_idx in downstream_edge_dict[node_idx]:
                model.add_constraint(shard_intermediate_buffer_size_depth_original[downstream_tensor_idx] * tiling_factor[node_idx] == shard_intermediate_buffer_size[i] * depth[i])
                model.add_constraint(shard_intermediate_buffer_size_depth_two[downstream_tensor_idx] * tiling_factor[node_idx] == shard_intermediate_buffer_size[i] * 2)
                tiled_edge_list.append(downstream_tensor_idx)
        
    for i in range(num_node):
        if i not in tiled_kernel_list:
            model.add_constraint(tiling_factor[i] == 1)

    for i in range(num_node):
        aaa = model.add_variable(vtype='binary')
        model.add_constraint((aaa == 1) >> (tiling_factor[i] >= 2))
        model.add_constraint((aaa == 0) >> (tiling_factor[i] == 1))
        model.add_constraint((aaa == 1) >> (Par_total[i] == Core))  # enforce the kernel takes an entire config

    for i in range(num_edge):
        if i not in tiled_edge_list:
            model.add_constraint(shard_intermediate_buffer_size_depth_original[i] == shard_intermediate_buffer_size[i] * depth[i])
            model.add_constraint(shard_intermediate_buffer_size_depth_two[i] == shard_intermediate_buffer_size[i] * 2)





    SRAM_Per_Config_extra = model.add_matrix_variable(C, name='SRAM_Per_Config_extra', vtype='continuous', lb=0)
    SRAM_Per_Config_total = model.add_matrix_variable(C, name='SRAM_Per_Config_total', vtype='continuous', lb=0)
    SRAM_Per_Config_intermediate_dram = model.add_matrix_variable(C, name='SRAM_Per_Config_intermediate_dram', vtype='continuous', lb=0)
    SRAM_Per_Config_intermediate_onchip = model.add_matrix_variable(C, name='SRAM_Per_Config_intermediate_onchip', vtype='continuous', lb=0)
    SRAM_Per_Config_initiation = model.add_matrix_variable(C, name='SRAM_Per_Config_initiation', vtype='continuous', lb=0)
    for i in range(C):
        model.add_constraint(SRAM_Per_Config_extra[i] == A[:, i] @ np.array(sram_extra))
        model.add_constraint(SRAM_Per_Config_intermediate_dram[i] == shard_intermediate_buffer_size_depth_two @ D[:, i])       
        model.add_constraint(SRAM_Per_Config_intermediate_onchip[i] == shard_intermediate_buffer_size_depth_original @ B[:, i])
        model.add_constraint(SRAM_Per_Config_initiation[i] == shard_initiation_buffer_size_depth_one @ F[:, i])
        model.add_constraint(SRAM_Per_Config_total[i] == SRAM_Per_Config_extra[i] + SRAM_Per_Config_intermediate_dram[i] + SRAM_Per_Config_intermediate_onchip[i] + SRAM_Per_Config_initiation[i])
        
        if dse.execution.WhichOneof('workload_variant') == 'llm':
            model.add_constraint(SRAM_Per_Config_total[i] <= sram_cap)








    tiling_per_config = model.add_matrix_variable(C, name='tiling_per_config', vtype='integer')
    for i in range(C):
        tmp = model.add_matrix_variable(num_node, vtype='integer')

        for j in range(num_node):
            model.add_constraint(tmp[j] == A[j, i] * tiling_factor[j])
        
        model.add_constraint(tiling_per_config[i] == highspy.max_(tmp[j] for j in range(num_node)))

    


else:
    p_and_r_flag = True
    
    # label lane/stage
    
    # input edge dict: map node idx to input edge idx
    # output edge dict: map node idx to output edge idx
    downstream_edge_dict = dict()
    for i in range(num_edge):
        start_node_idx = node_dict[startIdx[i]]
        end_node_idx = node_dict[endIdx[i]]
        
        if start_node_idx not in downstream_edge_dict.keys():
            downstream_edge_dict[start_node_idx] = [i]
        else:
            downstream_edge_dict[start_node_idx].append(i)
    

    upstream_edge_dict = dict()
    for i in range(num_edge):
        start_node_idx = node_dict[startIdx[i]]
        end_node_idx = node_dict[endIdx[i]]
        
        if end_node_idx not in upstream_edge_dict.keys():
            upstream_edge_dict[end_node_idx] = [i]
        else:
            upstream_edge_dict[end_node_idx].append(i)


    for i in range(num_node):
        if i in upstream_edge_dict.keys():
            if node_type[i] == 'gemm_input1_weight':
                upstream_tensor_idx = upstream_edge_dict[i][0]
                lane_stage_type[upstream_tensor_idx] = 'stage'
                
            elif node_type[i] == 'gemm_input1_input2':
                if len(upstream_edge_dict[i]) == 2:
                    upstream_tensor_idx_1 = upstream_edge_dict[i][0]
                    upstream_tensor_idx_2 = upstream_edge_dict[i][1]
                    lane_stage_type[upstream_tensor_idx_1] = 'lane'
                    lane_stage_type[upstream_tensor_idx_2] = 'stage'
                elif len(upstream_edge_dict[i]) == 1:
                    upstream_tensor_idx_1 = upstream_edge_dict[i][0]
                    lane_stage_type[upstream_tensor_idx_1] = 'lane'
                else:
                    raise Exception('Wrong!')
                
            elif node_type[i] == 'elementwise_input1':
                upstream_tensor_idx = upstream_edge_dict[i][0]
                lane_stage_type[upstream_tensor_idx] = 'lane'
                
            elif node_type[i] == 'elementwise_input1_input2':
                if len(upstream_edge_dict[i]) == 2:
                    upstream_tensor_idx_1 = upstream_edge_dict[i][0]
                    upstream_tensor_idx_2 = upstream_edge_dict[i][1]
                    lane_stage_type[upstream_tensor_idx_1] = 'lane'
                    lane_stage_type[upstream_tensor_idx_2] = 'lane'
                elif len(upstream_edge_dict[i]) == 1:
                    upstream_tensor_idx_1 = upstream_edge_dict[i][0]
                    lane_stage_type[upstream_tensor_idx_1] = 'lane'
                else:
                    raise Exception('Wrong!')
                
            else:
                raise Exception('Wrong!')

    

    
    
    # SRAM cap
    shard_intermediate_buffer_size_depth_original = model.add_matrix_variable(num_edge, name='shard_intermediate_buffer_size_depth_original', vtype='integer', lb=0)
    shard_intermediate_buffer_size_depth_two = model.add_matrix_variable(num_edge, name='shard_intermediate_buffer_size_depth_two', vtype='integer', lb=0)
    shard_initiation_buffer_size_depth_one = model.add_matrix_variable(num_weight, name='shard_initiation_buffer_size_depth_one', vtype='integer', lb=0)


    



    tiled_kernel_list = []
    tiled_edge_list = []
    tiling_factor = model.add_matrix_variable(num_node, name='tiling_factor', vtype='integer', lb=1)  
    for i in range(num_weight):
        node_idx = node_dict[weight_dict[i]]
        tiled_kernel_list.append(node_idx)
        
        model.add_constraint(shard_initiation_buffer_size_depth_one[i] * tiling_factor[node_idx] == shard_initiation_buffer_size[i] * 1)
        
        if node_idx in downstream_edge_dict.keys():
            for downstream_tensor_idx in downstream_edge_dict[node_idx]:
                model.add_constraint(shard_intermediate_buffer_size_depth_original[downstream_tensor_idx] * tiling_factor[node_idx] == shard_intermediate_buffer_size[i] * depth[i])
                model.add_constraint(shard_intermediate_buffer_size_depth_two[downstream_tensor_idx] * tiling_factor[node_idx] == shard_intermediate_buffer_size[i] * 2)
                tiled_edge_list.append(downstream_tensor_idx)
        
    for i in range(num_node):
        if i not in tiled_kernel_list:
            model.add_constraint(tiling_factor[i] == 1)

    for i in range(num_node):
        aaa = model.add_variable(vtype='binary')
        model.add_constraint((aaa == 1) >> (tiling_factor[i] >= 2))
        model.add_constraint((aaa == 0) >> (tiling_factor[i] == 1))
        model.add_constraint((aaa == 1) >> (Par_total[i] == Core))  # enforce the kernel takes an entire config

    for i in range(num_edge):
        if i not in tiled_edge_list:
            model.add_constraint(shard_intermediate_buffer_size_depth_original[i] == shard_intermediate_buffer_size[i] * depth[i])
            model.add_constraint(shard_intermediate_buffer_size_depth_two[i] == shard_intermediate_buffer_size[i] * 2)






    SRAM_Per_Config_total = model.add_matrix_variable(C, name='SRAM_Per_Config_total', vtype='continuous', lb=0)
    SRAM_Per_Config_intermediate_dram = model.add_matrix_variable(C, name='SRAM_Per_Config_intermediate_dram', vtype='continuous', lb=0)
    SRAM_Per_Config_intermediate_onchip = model.add_matrix_variable(C, name='SRAM_Per_Config_intermediate_onchip', vtype='continuous', lb=0)
    SRAM_Per_Config_initiation = model.add_matrix_variable(C, name='SRAM_Per_Config_initiation', vtype='continuous', lb=0)

    PMU_used_intermediate = model.add_matrix_variable((num_edge), name='PMU_used_intermediate', vtype='integer', lb=0)
    PMU_used_initiation = model.add_matrix_variable((num_weight), name='PMU_used_initiation', vtype='integer', lb=0)
    PMU_used_per_config = model.add_matrix_variable((C), name='PMU_used_per_config', vtype='integer', lb=0)
    for i in range(C):
        for j in range(num_edge):
            model.add_constraint(PMU_used_intermediate[j] * dse.system.accelerator.pmu_cap >= shard_intermediate_buffer_size_depth_two[j] * D[j, i])
            model.add_constraint(PMU_used_intermediate[j] * dse.system.accelerator.pmu_cap >= shard_intermediate_buffer_size_depth_original[j] * B[j, i])
                
            if lane_stage_type[j] == 'lane':
                model.add_constraint(PMU_used_intermediate[j] * LaneWidth >= Par_lane[node_dict[endIdx[j]]] * LaneWidth)
            elif lane_stage_type[j] == 'stage':
                model.add_constraint(PMU_used_intermediate[j] * LaneWidth >= Par_stage[node_dict[endIdx[j]]] * StageWidth)
            else:
                raise Exception('Wrong!')
        
        for j in range(num_weight):
            model.add_constraint(PMU_used_initiation[j] * dse.system.accelerator.pmu_cap >= shard_initiation_buffer_size_depth_one[j] * F[j, i])
            model.add_constraint(PMU_used_initiation[j] * LaneWidth >= Par_lane[node_dict[weight_dict[j]]] * LaneWidth)
        
    
        model.add_constraint(PMU_used_per_config[i] == PMU_used_intermediate @ Z[:, i] + PMU_used_initiation @ F[:, i])
        model.add_constraint(PMU_used_per_config[i] <= dse.system.accelerator.pmu)

    # downstream edge
    for i in range(num_node):
        if i in downstream_edge_dict.keys():
            for j in downstream_edge_dict[i]:
                if node_type[i] == 'gemm_input1_weight':
                    model.add_constraint(PMU_used_intermediate[j] * LaneWidth * shard_K[i] >= Par_lane[i] * Par_stage[i] * LaneWidth * StageWidth)
                    
                elif node_type[i] == 'gemm_input1_input2':
                    model.add_constraint(PMU_used_intermediate[j] * LaneWidth * shard_K[i] >= Par_lane[i] * Par_stage[i] * LaneWidth * StageWidth)
                    
                elif node_type[i] == 'elementwise_input1':
                    model.add_constraint(PMU_used_intermediate[j] * LaneWidth >= Par_lane[i] * LaneWidth)
                    
                elif node_type[i] == 'elementwise_input1_input2':
                    model.add_constraint(PMU_used_intermediate[j] * LaneWidth >= Par_lane[i] * LaneWidth)
                    
                else:
                    raise Exception('Wrong!')




    tiling_per_config = model.add_matrix_variable(C, name='tiling_per_config', vtype='integer')
    for i in range(C):
        model.add_constraint(tiling_per_config[i] == A[:, i] @ tiling_factor)












    
    
# dram cap
dram_bytes_per_config_intermediate = model.add_matrix_variable(C, name='dram_bytes_per_config_intermediate', vtype='continuous', lb=0)
dram_bytes_per_config_initiation = model.add_matrix_variable(C, name='dram_bytes_per_config_initiation', vtype='continuous', lb=0)
for i in range(C):
    aaa = model.add_variable(vtype='continuous')
    # model.add_constraint(aaa == (shard_intermediate_buffer_size @ D[:, i]))
    model.add_constraint(aaa == (shard_intermediate_buffer_size @ E[:, i]))
    model.add_constraint(dram_bytes_per_config_intermediate[i] == aaa * num_tile_per_config[i])
    model.add_constraint(dram_bytes_per_config_initiation[i] == shard_initiation_buffer_size @ F[:, i])







dram_bytes_extra = model.add_variable(name='dram_bytes_extra', vtype='continuous', lb=0)
model.add_constraint(dram_bytes_extra == sum(dram_extra) * layer_per_stage)


dram_bytes_initiation = model.add_variable(name='dram_bytes_initiation', vtype='continuous', lb=0)
dram_bytes_intermediate = model.add_variable(name='dram_bytes_intermediate', vtype='continuous', lb=0)
model.add_constraint(dram_bytes_initiation == np.ones((C)) @ dram_bytes_per_config_initiation)
model.add_constraint(dram_bytes_intermediate == highspy.max_(dram_bytes_per_config_intermediate[j] for j in range(C)))
dram_bytes_total = model.add_variable(name='dram_bytes_total', vtype='continuous', lb=0)

if dse.execution.WhichOneof('workload_variant') == 'llm':
    weight = model.add_variable(name='weight', vtype='continuous', lb=0)
    activation = model.add_variable(name='activation', vtype='continuous', lb=0)
    
    model.add_constraint(weight == dram_bytes_initiation * layer_per_stage)
    model.add_constraint(activation == dram_bytes_intermediate * Micro_Batch_Size)
    model.add_constraint(dram_bytes_total == weight + activation + dram_bytes_extra)

elif dse.execution.WhichOneof('workload_variant') == 'hpl':
    model.add_constraint(dram_bytes_total == HPL_DRAM_SIZE + dram_bytes_extra)

elif dse.execution.WhichOneof('workload_variant') == 'fft':
    model.add_constraint(dram_bytes_total == FFT_DRAM_SIZE + dram_bytes_extra)

elif dse.execution.WhichOneof('workload_variant') == 'dlrm':
    weight = model.add_variable(name='weight', vtype='continuous', lb=0)
    activation = model.add_variable(name='activation', vtype='continuous', lb=0)

    model.add_constraint(weight == dram_bytes_initiation * layer_per_stage)
    model.add_constraint(activation == dram_bytes_intermediate * Micro_Batch_Size)
    model.add_constraint(dram_bytes_total == weight + activation + sharded_table_size + dram_bytes_extra)

else:
    raise Exception('Wrong!')
    


    
model.add_constraint(dram_bytes_total <= DRAM_Cap)  
DRAM_BW = model.add_variable(name='DRAM_BW', vtype='continuous', lb=0)
model.add_constraint(DRAM_BW == dse.system.memory.dram_bw)


# compute cycle
if dse.execution.compute_util == 0:
    Cycle = model.add_matrix_variable(num_node, name='Cycle', vtype='integer', lb=0)
    m_factor = model.add_matrix_variable(num_node, name='m_factor', vtype='integer', lb=1)
    n_factor = model.add_matrix_variable(num_node, name='n_factor', vtype='integer', lb=1)
    
    MMM = model.add_matrix_variable(num_node, name='MMM', vtype='integer', lb=1)
    KKK = model.add_matrix_variable(num_node, name='KKK', vtype='integer', lb=1)
    NNN = model.add_matrix_variable(num_node, name='NNN', vtype='integer', lb=1)
    
    SIMD_or_Systolic = []

else:
    FLOP_per_kernel = model.add_matrix_variable(num_node, name='FLOP_per_kernel', vtype='integer', lb=0)


for i in range(num_node):
    if kernel_type[i] == KernelType.SIMD.value:
        if dse.execution.compute_util == 0:
            model.add_constraint(m_factor[i] * Par_lane[i] * LaneWidth >= shard_M[i])
            model.add_constraint(Par_stage[i] == 1)
            model.add_constraint(Cycle[i] == m_factor[i] * shard_N[i])
            
            model.add_constraint(MMM[i] == m_factor[i])
            model.add_constraint(KKK[i] == 1)
            model.add_constraint(NNN[i] == shard_N[i])
            
            SIMD_or_Systolic.append('SIMD')

        else:
            model.add_constraint(FLOP_per_kernel[i] == shard_M[i] * shard_N[i])
        
    elif kernel_type[i] == KernelType.SYSTOLIC.value:
        if dse.execution.compute_util == 0:
            if use_effective_stage[i]:
                model.add_constraint(m_factor[i] * Par_lane[i] * LaneWidth >= shard_M[i])
                model.add_constraint(n_factor[i] * Par_stage[i] * dse.execution.effective_stage >= shard_K[i])

                tmp = model.add_variable(vtype='integer', lb=0)
                model.add_constraint(tmp == m_factor[i] * n_factor[i])
                model.add_constraint(Cycle[i] == tmp * shard_N[i])
            
            else:
                model.add_constraint(m_factor[i] * Par_lane[i] * LaneWidth >= shard_M[i])
                model.add_constraint(n_factor[i] * Par_stage[i] * StageWidth >= shard_N[i])

                tmp = model.add_variable(vtype='integer', lb=0)
                model.add_constraint(tmp == m_factor[i] * n_factor[i])
                model.add_constraint(Cycle[i] == tmp * shard_K[i])
            
            model.add_constraint(MMM[i] == m_factor[i])
            model.add_constraint(KKK[i] == shard_K[i])
            model.add_constraint(NNN[i] == n_factor[i])
            
            SIMD_or_Systolic.append('Systolic')
            
        else:
            aaa = model.add_variable(vtype='integer', lb=0)
            model.add_constraint(aaa == 2 * shard_M[i] * shard_N[i])
            model.add_constraint(FLOP_per_kernel[i] == aaa * shard_K[i])
            
    else:
        raise Exception('Wrong!')
    

Compute_Latency = model.add_matrix_variable(C, name='Compute_Latency', vtype='continuous', lb=0)
for i in range(C):
    if dse.execution.compute_util == 0:
        t1 = model.add_matrix_variable(num_node, vtype='integer', lb=0)
        t2 = model.add_variable(vtype='continuous')
        for j in range(num_node):
            model.add_constraint(t1[j] == Cycle[j] * A[j, i])
        model.add_constraint(t2 == highspy.max_(t1[j] for j in range(num_node)))

        model.add_constraint(Compute_Latency[i] == t2 * num_tile_per_config[i] / Freq)
        
    else:
        t3 = model.add_matrix_variable(num_node, vtype='integer', lb=0)
        t4 = model.add_variable(vtype='continuous')
        for j in range(num_node):
            model.add_constraint(t3[j] == FLOP_per_kernel[j] * A[j, i])
        model.add_constraint(t4 == t3 @ np.ones((num_node)))
        
        tmp = 1 / dse.execution.compute_util
        model.add_constraint(Compute_Latency[i] == t4 * num_tile_per_config[i] * tmp / GFLOPS)





Memory_Latency = model.add_matrix_variable(C, name='Memory_Latency', vtype='continuous', lb=0)
memory_latency = model.add_matrix_variable(C, name='memory_latency', vtype='continuous', lb=0)
explicit_memory_latency = model.add_matrix_variable(C, name='explicit_memory_latency', vtype='continuous', lb=0)
for i in range(C):
    t1 = model.add_variable(vtype='continuous')
    t2 = model.add_variable(vtype='continuous')
    model.add_constraint(t1 == shard_intermediate_buffer_size @ D[:, i])
    model.add_constraint(t2 == num_tile_per_config[i] * tiling_per_config[i])

    model.add_constraint(memory_latency[i] * DRAM_BW == t1 * t2)
    model.add_constraint(explicit_memory_latency[i] * DRAM_BW == memory_size @ A[:, i])

    model.add_constraint(Memory_Latency[i] == memory_latency[i] + explicit_memory_latency[i])

















    
    
        
    




Network_Latency = model.add_matrix_variable(C, name='Network_Latency', vtype='continuous', lb=0)
p2p_latency = model.add_variable(name='p2p_latency', vtype='continuous')

if dse.execution.WhichOneof('workload_variant') == 'llm':


    # TP
    Network_Latency_ALL_REDUCE_node = model.add_matrix_variable(C, name='Network_Latency_ALL_REDUCE_node', vtype='continuous', lb=0)
    if is_TP_hierarchical == True:
        for i in range(C):
            t1 = model.add_variable(vtype='continuous')
            t2 = model.add_variable(vtype='continuous')
            t3 = model.add_variable(vtype='continuous')
            model.add_constraint(t1 == A[:, i] @ ALL_REDUCE_communication_size_node)
            model.add_constraint(t2 == ALL_REDUCE_ratio_fast * num_tile_per_config[i])
            model.add_constraint(t3 == ALL_REDUCE_ratio_slow * num_tile_per_config[i])
            model.add_constraint(Network_Latency_ALL_REDUCE_node[i] == t1*t2 + t1*t3) # reduce-scatter/all-gather
        
    else:  
        t3 = model.add_matrix_variable((num_node, C), vtype='continuous')
        t4 = model.add_matrix_variable((num_node, C), vtype='binary')
        t5 = model.add_matrix_variable(C, vtype='binary')

        link_latency_allreduce_node = model.add_matrix_variable(C, name='link_latency_allreduce_node', vtype='continuous')
        serialization_latency_allreduce_node = model.add_matrix_variable(C, name='serialization_latency_allreduce_node', vtype='continuous')

        for i in range(C):
            t1 = model.add_variable(vtype='continuous')
            t2 = model.add_variable(vtype='continuous')

            model.add_constraint(t1 == A[:, i] @ ALL_REDUCE_communication_size_node)
            model.add_constraint(t2 == ALL_REDUCE_ratio * num_tile_per_config[i])
            
            for j in range(num_node):
                model.add_constraint(t3[j, i] == A[j, i] * ALL_REDUCE_communication_size_node[j])

                aaa = model.add_variable(vtype='binary')
                model.add_constraint((aaa == 1) >> (t3[j, i] >= 1))
                model.add_constraint((aaa == 0) >> (t3[j, i] <= 0))
                model.add_constraint((aaa == 1) >> (t4[j, i] == 1))
                model.add_constraint((aaa == 0) >> (t4[j, i] == 0))

            
            if 'TP' in par:
                if dse.execution.separate_rs_ag_for_ar:
                    tmp = 2
                else:
                    tmp = 1

                if topology[par.index('TP')] == BasicTopology.R.value:
                    model.add_constraint(link_latency_allreduce_node[i] == t4[:, i].sum() * dse.system.accelerator.link_latency * tmp * (dimension[par.index('TP')]-1))
                elif topology[par.index('TP')] == BasicTopology.SW.value:
                    model.add_constraint(link_latency_allreduce_node[i] == t4[:, i].sum() * dse.system.accelerator.link_latency * tmp * (dimension[par.index('TP')]-1))
                elif topology[par.index('TP')] == BasicTopology.FC.value:
                    model.add_constraint(link_latency_allreduce_node[i] == t4[:, i].sum() * dse.system.accelerator.link_latency * tmp)
            else:
                model.add_constraint(link_latency_allreduce_node[i] == 0)
            
            model.add_constraint(serialization_latency_allreduce_node[i] == t1*t2)
            model.add_constraint(Network_Latency_ALL_REDUCE_node[i] == serialization_latency_allreduce_node[i] + link_latency_allreduce_node[i])

            
    # TP
    Network_Latency_ALL_TO_ALL_node = model.add_matrix_variable(C, name='Network_Latency_ALL_TO_ALL_node', vtype='continuous', lb=0)
    for i in range(C):
        t1 = model.add_variable(vtype='continuous')
        t2 = model.add_variable(vtype='continuous')
        model.add_constraint(t1 == A[:, i] @ ALL_TO_ALL_communication_size_node)
        model.add_constraint(t2 == ALL_TO_ALL_ratio * num_tile_per_config[i])
        model.add_constraint(Network_Latency_ALL_TO_ALL_node[i] == t1*t2)

    # TP
    Network_Latency_ALL_GATHER_node = model.add_matrix_variable(C, name='Network_Latency_ALL_GATHER_node', vtype='continuous', lb=0)
    for i in range(C):
        t1 = model.add_variable(vtype='continuous')
        t2 = model.add_variable(vtype='continuous')
        model.add_constraint(t1 == A[:, i] @ ALL_GATHER_communication_size_node)
        model.add_constraint(t2 == ALL_GATHER_ratio * num_tile_per_config[i])
        model.add_constraint(Network_Latency_ALL_GATHER_node[i] == t1*t2)


    # TP
    Network_Latency_ALL_REDUCE_edge = model.add_matrix_variable(C, name='Network_Latency_ALL_REDUCE_edge', vtype='continuous', lb=0)
    t3 = model.add_matrix_variable((num_edge, C), vtype='continuous')
    t4 = model.add_matrix_variable((num_edge, C), vtype='binary')
    t5 = model.add_matrix_variable(C, vtype='binary')

    link_latency_allreduce_edge = model.add_matrix_variable(C, name='link_latency_allreduce_edge', vtype='continuous')
    serialization_latency_allreduce_edge = model.add_matrix_variable(C, name='serialization_latency_allreduce_edge', vtype='continuous')

    for i in range(C):
        t1 = model.add_variable(vtype='continuous')
        t2 = model.add_variable(vtype='continuous')
        
        model.add_constraint(t1 == H[:, i] @ ALL_REDUCE_communication_size_edge)
        model.add_constraint(t2 == ALL_REDUCE_ratio * num_tile_per_config[i])

        for j in range(num_edge):
            model.add_constraint(t3[j, i] == H[j, i] * ALL_REDUCE_communication_size_edge[j])

            aaa = model.add_variable(vtype='binary')
            model.add_constraint((aaa == 1) >> (t3[j, i] >= 1))
            model.add_constraint((aaa == 0) >> (t3[j, i] <= 0))
            model.add_constraint((aaa == 1) >> (t4[j, i] == 1))
            model.add_constraint((aaa == 0) >> (t4[j, i] == 0))

        if 'TP' in par:
            if dse.execution.separate_rs_ag_for_ar:
                tmp = 2
            else:
                tmp = 1

            if topology[par.index('TP')] == BasicTopology.R.value:
                model.add_constraint(link_latency_allreduce_edge[i] == t4[:, i].sum() * dse.system.accelerator.link_latency * tmp * (dimension[par.index('TP')]-1))
            elif topology[par.index('TP')] == BasicTopology.SW.value:
                model.add_constraint(link_latency_allreduce_edge[i] == t4[:, i].sum() * dse.system.accelerator.link_latency * tmp * (dimension[par.index('TP')]-1))
            elif topology[par.index('TP')] == BasicTopology.FC.value:
                model.add_constraint(link_latency_allreduce_edge[i] == t4[:, i].sum() * dse.system.accelerator.link_latency * tmp)
        else:
            model.add_constraint(link_latency_allreduce_edge[i] == 0)

        model.add_constraint(serialization_latency_allreduce_edge[i] == t1*t2)
        model.add_constraint(Network_Latency_ALL_REDUCE_edge[i] == serialization_latency_allreduce_edge[i] + link_latency_allreduce_edge[i])
        



    # TP
    Network_Latency_ALL_TO_ALL_edge = model.add_matrix_variable(C, name='Network_Latency_ALL_TO_ALL_edge', vtype='continuous', lb=0)
    for i in range(C):
        t1 = model.add_variable(vtype='continuous')
        t2 = model.add_variable(vtype='continuous')
        model.add_constraint(t1 == H[:, i] @ ALL_TO_ALL_communication_size_edge)
        model.add_constraint(t2 == ALL_TO_ALL_ratio * num_tile_per_config[i])
        model.add_constraint(Network_Latency_ALL_TO_ALL_edge[i] == t1*t2)




    # TP
    Network_Latency_ALL_GATHER_edge = model.add_matrix_variable(C, name='Network_Latency_ALL_GATHER_edge', vtype='continuous', lb=0)
    for i in range(C):
        t1 = model.add_variable(vtype='continuous')
        t2 = model.add_variable(vtype='continuous')
        model.add_constraint(t1 == H[:, i] @ ALL_GATHER_communication_size_edge)
        model.add_constraint(t2 == ALL_GATHER_ratio * num_tile_per_config[i])
        model.add_constraint(Network_Latency_ALL_GATHER_edge[i] == t1*t2)


















    # DP
    Network_Latency_ALL_REDUCE_PERIODIC_node = model.add_matrix_variable(C, name='Network_Latency_ALL_REDUCE_PERIODIC_node', vtype='continuous', lb=0)
    if is_DP_hierarchical == True:
        for i in range(C):
            t1 = model.add_variable(vtype='continuous')
            t2 = model.add_variable(vtype='continuous')
            t3 = model.add_variable(vtype='continuous')
            model.add_constraint(t1 == A[:, i] @ ALL_REDUCE_PERIODIC_communication_size_node)
            model.add_constraint(t2 == ALL_REDUCE_PERIODIC_ratio_fast)
            model.add_constraint(t3 == ALL_REDUCE_PERIODIC_ratio_slow)
            model.add_constraint(Network_Latency_ALL_REDUCE_PERIODIC_node[i] == t1*t2 + t1*t3)
            
    else:
        for i in range(C):
            t1 = model.add_variable(vtype='continuous')
            t2 = model.add_variable(vtype='continuous')
            model.add_constraint(t1 == A[:, i] @ ALL_REDUCE_PERIODIC_communication_size_node)
            model.add_constraint(t2 == ALL_REDUCE_PERIODIC_ratio)
            model.add_constraint(Network_Latency_ALL_REDUCE_PERIODIC_node[i] == t1*t2) # reduce-scatter/all-gather
















    # SP
    Network_Latency_ALL_REDUCE_node_sp = model.add_matrix_variable(C, name='Network_Latency_ALL_REDUCE_node_sp', vtype='continuous', lb=0)
    for i in range(C):
        t1 = model.add_variable(vtype='continuous')
        t2 = model.add_variable(vtype='continuous')
        model.add_constraint(t1 == A[:, i] @ ALL_REDUCE_communication_size_node_sp)
        model.add_constraint(t2 == ALL_REDUCE_ratio_sp * num_tile_per_config[i])       
        model.add_constraint(Network_Latency_ALL_REDUCE_node_sp[i] == t1*t2)

        





    # SP
    Network_Latency_ALL_GATHER_node_sp = model.add_matrix_variable(C, name='Network_Latency_ALL_GATHER_node_sp', vtype='continuous', lb=0)
    for i in range(C):
        t1 = model.add_variable(vtype='continuous')
        t2 = model.add_variable(vtype='continuous')
        model.add_constraint(t1 == A[:, i] @ ALL_GATHER_communication_size_node_sp)
        model.add_constraint(t2 == ALL_GATHER_ratio_sp * num_tile_per_config[i])
        model.add_constraint(Network_Latency_ALL_GATHER_node_sp[i] == t1*t2)



    # SP
    Network_Latency_ALL_REDUCE_edge_sp = model.add_matrix_variable(C, name='Network_Latency_ALL_REDUCE_edge_sp', vtype='continuous', lb=0)
    for i in range(C):
        t1 = model.add_variable(vtype='continuous')
        t2 = model.add_variable(vtype='continuous')
        
        model.add_constraint(t1 == H[:, i] @ ALL_REDUCE_communication_size_edge_sp)
        model.add_constraint(t2 == ALL_REDUCE_ratio_sp * num_tile_per_config[i])
        model.add_constraint(Network_Latency_ALL_REDUCE_edge_sp[i] == t1*t2)





    # SP
    Network_Latency_ALL_GATHER_edge_sp = model.add_matrix_variable(C, name='Network_Latency_ALL_GATHER_edge_sp', vtype='continuous', lb=0)
    for i in range(C):
        t1 = model.add_variable(vtype='continuous')
        t2 = model.add_variable(vtype='continuous')
        model.add_constraint(t1 == H[:, i] @ ALL_GATHER_communication_size_edge_sp)
        model.add_constraint(t2 == ALL_GATHER_ratio_sp * num_tile_per_config[i])
        model.add_constraint(Network_Latency_ALL_GATHER_edge_sp[i] == t1*t2)










    for i in range(C):
        aaa = model.add_variable(vtype='continuous')
        bbb = model.add_variable(vtype='continuous')
        ccc = model.add_variable(vtype='continuous')
        model.add_constraint(aaa == Network_Latency_ALL_REDUCE_node[i]
                             + Network_Latency_ALL_TO_ALL_node[i]
                             + Network_Latency_ALL_GATHER_node[i]
                             + Network_Latency_ALL_REDUCE_edge[i]
                             + Network_Latency_ALL_TO_ALL_edge[i]
                             + Network_Latency_ALL_GATHER_edge[i])
        model.add_constraint(bbb == Network_Latency_ALL_REDUCE_PERIODIC_node[i])
        model.add_constraint(ccc == Network_Latency_ALL_REDUCE_node_sp[i]
                             + Network_Latency_ALL_GATHER_node_sp[i]
                             + Network_Latency_ALL_REDUCE_edge_sp[i]
                             + Network_Latency_ALL_GATHER_edge_sp[i])
        
        model.add_constraint(Network_Latency[i] == highspy.max_(aaa, bbb, ccc))

    

    # p2p communication
    model.add_constraint(p2p_latency == P2P_ratio * Intermediate * layers)
    

elif dse.execution.WhichOneof('workload_variant') == 'dlrm':
    # A2A
    Network_Latency_ALL_TO_ALL_tmp = model.add_matrix_variable((C, len(topology)), name='Network_Latency_ALL_TO_ALL_tmp', vtype='continuous', lb=0)
    Network_Latency_ALL_TO_ALL = model.add_matrix_variable(C, name='Network_Latency_ALL_TO_ALL', vtype='continuous', lb=0)

    for i in range(C):
        for j in range(len(topology)):
            aaa = model.add_variable(vtype='continuous')
            model.add_constraint(aaa == A[:, i] @ ALL_TO_ALL_communication_size_node) # per-chip bytes
            
            model.add_constraint(Network_Latency_ALL_TO_ALL_tmp[i, j] * Link_BW[j] * a2a_bw_factor[j] >= aaa * a2a_msg_factor[j])
    
    for i in range(C):
        model.add_constraint(Network_Latency_ALL_TO_ALL[i] == highspy.max_(Network_Latency_ALL_TO_ALL_tmp[i, j] for j in range(len(topology))))



        


    # data parallelism all-reduce
    Network_Latency_ALL_REDUCE_PERIODIC = model.add_matrix_variable(C, name='Network_Latency_ALL_REDUCE_PERIODIC', vtype='continuous', lb=0)
    for i in range(C):
        model.add_constraint(Network_Latency_ALL_REDUCE_PERIODIC[i] >= A[:, i] @ ALL_REDUCE_PERIODIC_communication_size_node) # reduce-scatter/all-gather

        

    # total network latency
    for i in range(C):
        model.add_constraint(Network_Latency[i] == Network_Latency_ALL_TO_ALL[i] + Network_Latency_ALL_REDUCE_PERIODIC[i])
    
    model.add_constraint(p2p_latency == 0)
    
    
    
    
    
    
    
    
elif dse.execution.WhichOneof('workload_variant') == 'hpl':
    Network_Latency_POINT_TO_POINT = model.add_matrix_variable(C, name='Network_Latency_POINT_TO_POINT', vtype='continuous', lb=0)
    Network_Latency_BROADCAST = model.add_matrix_variable(C, name='Network_Latency_BROADCAST', vtype='continuous', lb=0)
    
    for i in range(C):
        # X dim
        aaa = model.add_variable(vtype='continuous')
        model.add_constraint(aaa == A[:, i] @ POINT_TO_POINT_communication_size)
        
        # Y dim
        bbb = model.add_variable(vtype='continuous')
        model.add_constraint(bbb == A[:, i] @ BROADCAST_communication_size)
        
        model.add_constraint(Network_Latency_POINT_TO_POINT[i] * Link_BW[0] >= aaa)
        model.add_constraint(Network_Latency_BROADCAST[i] * Link_BW[1] >= bbb)
        
    for i in range(C):
        model.add_constraint(Network_Latency[i] == highspy.max_(Network_Latency_POINT_TO_POINT[i], Network_Latency_BROADCAST[i]))  
    
    model.add_constraint(p2p_latency == 0)
    
    
    
    
elif dse.execution.WhichOneof('workload_variant') == 'fft':
    # A2A
    Network_Latency_ALL_TO_ALL_tmp = model.add_matrix_variable((C, len(topology)), name='Network_Latency_ALL_TO_ALL_tmp', vtype='continuous', lb=0)
    Network_Latency_ALL_TO_ALL = model.add_matrix_variable(C, name='Network_Latency_ALL_TO_ALL', vtype='continuous', lb=0)
    
    for i in range(C):
        for j in range(len(topology)):
            aaa = model.add_variable(vtype='continuous')
            model.add_constraint(aaa == A[:, i] @ ALL_TO_ALL_communication_size_node) # per-chip bytes
            
            model.add_constraint(Network_Latency_ALL_TO_ALL_tmp[i, j] * Link_BW[j] * a2a_bw_factor[j] >= aaa * a2a_msg_factor[j])
    
    for i in range(C):
        model.add_constraint(Network_Latency_ALL_TO_ALL[i] == highspy.max_(Network_Latency_ALL_TO_ALL_tmp[i, j] for j in range(len(topology))))

    # total network latency
    for i in range(C):
        model.add_constraint(Network_Latency[i] == Network_Latency_ALL_TO_ALL[i])
        
    
    
    model.add_constraint(p2p_latency == 0)
    
    
 
else:
    raise Exception('Wrong!')



Compute_Memory_Latency = model.add_matrix_variable(C, name='Compute_Memory_Latency', vtype='continuous', lb=0)
Per_Config_II = model.add_matrix_variable(C, name='Per_Config_II', vtype='continuous', lb=0)
for i in range(C):
    if dse.execution.overlap == Overlap_Style.PERFECT_OVERLAP.value:
        tmp = model.add_variable(vtype='continuous')
        model.add_constraint(tmp == highspy.max_(Compute_Latency[i], Memory_Latency[i], Network_Latency[i]))
        model.add_constraint(Per_Config_II[i] == tmp * num_input_per_config[i])

    elif dse.execution.overlap == Overlap_Style.OVERLAP_COMPUTE_MEMORY.value:
        tmp = model.add_variable(vtype='continuous')
        model.add_constraint(Compute_Memory_Latency[i] == highspy.max_(Compute_Latency[i], Memory_Latency[i]))
        model.add_constraint(tmp == Compute_Memory_Latency[i] + Network_Latency[i])
        model.add_constraint(Per_Config_II[i] == tmp * num_input_per_config[i])

    elif dse.execution.overlap == Overlap_Style.NO_OVERLAP.value:
        
        if dse.execution.optimization == "" or dse.execution.optimization == "CMN":
            tmp = model.add_variable(vtype='continuous')
            model.add_constraint(tmp == Compute_Latency[i] + Memory_Latency[i] + Network_Latency[i])
            model.add_constraint(Per_Config_II[i] == tmp * num_input_per_config[i])
            
        elif dse.execution.optimization == "CM":
            tmp = model.add_variable(vtype='continuous')
            model.add_constraint(tmp == Compute_Latency[i] + Memory_Latency[i])
            model.add_constraint(Per_Config_II[i] == tmp * num_input_per_config[i])

        elif dse.execution.optimization == "CN":
            tmp = model.add_variable(vtype='continuous')
            model.add_constraint(tmp == Compute_Latency[i] + Network_Latency[i])
            model.add_constraint(Per_Config_II[i] == tmp * num_input_per_config[i])

        elif dse.execution.optimization == "C":
            tmp = model.add_variable(vtype='continuous')
            model.add_constraint(tmp == Compute_Latency[i])
            model.add_constraint(Per_Config_II[i] == tmp * num_input_per_config[i])
            
        else:
            raise Exception('Wrong!')
        



    else:
        raise Exception("Wrong!")
    

hhhh = model.add_variable(vtype='continuous', lb=0)
ns_per_batch = model.add_variable(name='ns_per_batch', vtype='continuous', lb=0)

if dse.execution.WhichOneof('workload_variant') == 'llm' or dse.execution.WhichOneof('workload_variant') == 'dlrm':
    all_config_II = model.add_variable(name='all_config_II', vtype='continuous', lb=0)
    tmp2 = model.add_variable(vtype='continuous')
    tmp3 = model.add_variable(vtype='continuous')
    tmp4 = model.add_variable(vtype='continuous')
    model.add_constraint(all_config_II == np.ones((C)) @ Per_Config_II)
    # model.add_constraint(all_config_II == total_compute_latency + total_network_latency + total_memory_latency)
    
    aaa = model.add_variable(vtype='continuous')
    pipeline_factor = model.add_variable(name='pipeline_factor', vtype='continuous')
    ccc = model.add_variable(vtype='continuous')
    ddd = model.add_variable(vtype='continuous')
    zzz = model.add_variable(vtype='binary')
    
    model.add_constraint((zzz == 0) >> (layer_per_stage >= 2))
    model.add_constraint((zzz == 1) >> (layer_per_stage == 1))
    model.add_constraint((zzz == 0) >> (ccc == layer_per_stage - 1))
    model.add_constraint((zzz == 1) >> (ccc == layer_per_stage))
    
    model.add_constraint(ddd * num_micro_batch_per_pipeline == PP - 1)
    model.add_constraint(ddd == aaa * ccc)
    model.add_constraint(pipeline_factor == aaa + 1)
    model.add_constraint(tmp2 == all_config_II * pipeline_factor)
    model.add_constraint(tmp3 == tmp2 * layer_per_stage)
    model.add_constraint(tmp4 * DP == tmp3 * global_batch_size)
    model.add_constraint(hhhh == tmp4)

    # inference
    if first_bwd_kernel == -1 and dse.execution.WhichOneof('workload_variant') == 'llm':
        final_latency_ns = model.add_variable(name='final_latency_ns', vtype='continuous')
        p2p_inference_size = model.add_variable(name='p2p_inference_size', vtype='continuous')
        vvv = model.add_variable(name='vvv', vtype='continuous')
        vvv1 = model.add_variable(name='vvv1', vtype='continuous')
        vvv2 = model.add_variable(name='vvv2', vtype='continuous')
        vvv3 = model.add_variable(name='vvv3', vtype='continuous')

        model.add_constraint(p2p_inference_size * TP >= hidden_dim * seq_len * word)

        model.add_constraint(vvv == all_config_II * layer_per_stage)
        model.add_constraint(vvv1 == vvv * PP)
        model.add_constraint(vvv2 * Link_BW_PP >= (PP-1) * p2p_inference_size)
        model.add_constraint(vvv3 == (PP-1) * dse.system.accelerator.link_latency)
        
        model.add_constraint(final_latency_ns == vvv1 + vvv2 + vvv3)

else:
    model.add_constraint(hhhh == np.ones((C)) @ Per_Config_II)

model.add_constraint(ns_per_batch == hhhh + p2p_latency)





# cost 
LINK_cost = model.add_matrix_variable(len(topology), name='LINK_cost', vtype='continuous', lb=0)
SWITCH_cost = model.add_matrix_variable(len(topology), name='SWITCH_cost', vtype='continuous', lb=0)

if dse.system.WhichOneof('topology_variant') == 'sw': # switch
    model.add_constraint(LINK_cost[0] == Shape[0] * Link_BW[0] * link_unit_price)
    model.add_constraint(SWITCH_cost[0] == Shape[0] * Link_BW[0] * switch_unit_price)
    
elif dse.system.WhichOneof('topology_variant') == 'fc': # fc
    aaa = model.add_variable(vtype='continuous')
    model.add_constraint(aaa == Shape[0] * (Shape[0] - 1))
    
    model.add_constraint(LINK_cost[0] == aaa * Link_BW[0] * link_unit_price)
    model.add_constraint(SWITCH_cost[0] == 0)
    
elif dse.system.WhichOneof('topology_variant') == 'r': # ring
    model.add_constraint(LINK_cost[0] == Shape[0] * Link_BW[0] * link_unit_price)
    model.add_constraint(SWITCH_cost[0] == 2 * Link_BW[0] * switch_unit_price)
    
elif dse.system.WhichOneof('topology_variant') == 'r_r': # 2d torus
    model.add_constraint(LINK_cost[0] == num_chips_per_copy * Link_BW[0] * link_unit_price)
    model.add_constraint(LINK_cost[1] == num_chips_per_copy * Link_BW[1] * link_unit_price)
    
    model.add_constraint(SWITCH_cost[0] == 2 * Shape[1] * Link_BW[0] * switch_unit_price)
    model.add_constraint(SWITCH_cost[1] == 2 * Shape[0] * Link_BW[1] * switch_unit_price)  
    
elif dse.system.WhichOneof('topology_variant') == 'fc_fc': # 2d dragonfly
    aaa = model.add_variable(vtype='continuous')
    bbb = model.add_variable(vtype='continuous')
    model.add_constraint(aaa == Shape[0] * (Shape[0] - 1))
    model.add_constraint(bbb == Shape[1] * (Shape[1] - 1))
    
    ccc = model.add_variable(vtype='continuous')
    model.add_constraint(ccc == aaa * Shape[1])
    model.add_constraint(LINK_cost[0] == ccc * Link_BW[0] * link_unit_price)
    model.add_constraint(LINK_cost[1] == bbb * Link_BW[1] * link_unit_price)
    
    ddd = model.add_variable(vtype='continuous')
    model.add_constraint(ddd == Shape[1] * Shape[1])
    model.add_constraint(SWITCH_cost[0] == 0)
    model.add_constraint(SWITCH_cost[1] == ddd * Link_BW[1] * switch_unit_price)
    
elif dse.system.WhichOneof('topology_variant') == 'r_fc': # zionex
    bbb = model.add_variable(vtype='continuous')
    model.add_constraint(bbb == Shape[1] * (Shape[1] - 1))
    
    model.add_constraint(LINK_cost[0] == num_chips_per_copy * Link_BW[0] * link_unit_price)
    model.add_constraint(LINK_cost[1] == bbb * Link_BW[1] * link_unit_price)
    
    ddd = model.add_variable(vtype='continuous')
    model.add_constraint(ddd == Shape[1] * Shape[1])
    model.add_constraint(SWITCH_cost[0] == 2 * Shape[1] * Link_BW[0] * switch_unit_price)
    model.add_constraint(SWITCH_cost[1] == ddd * Link_BW[1] * switch_unit_price)

elif dse.system.WhichOneof('topology_variant') == 'r_sw': # 2d dgx-1
        model.add_constraint(LINK_cost[0] == num_chip * Link_BW[0] * link_unit_price)
        model.add_constraint(LINK_cost[1] == Shape[1] * Link_BW[1] * link_unit_price)
        
        model.add_constraint(SWITCH_cost[0] == 2 * Shape[1] * Link_BW[0] * switch_unit_price)
        model.add_constraint(SWITCH_cost[1] == Shape[1] * Link_BW[1] * switch_unit_price)    
        
elif dse.system.WhichOneof('topology_variant') == 'sw_sw': # 2d dgx-2
    model.add_constraint(LINK_cost[0] == num_chip * Link_BW[0] * link_unit_price)
    model.add_constraint(LINK_cost[1] == Shape[1] * Link_BW[1] * link_unit_price)
    
    model.add_constraint(SWITCH_cost[0] == num_chip * Link_BW[0] * switch_unit_price)
    model.add_constraint(SWITCH_cost[1] == Shape[1] * Link_BW[1] * switch_unit_price)    
    
elif dse.system.WhichOneof('topology_variant') == 'r_r_r': # 3d torus
    aaa = model.add_variable(vtype='continuous')
    bbb = model.add_variable(vtype='continuous')
    ccc = model.add_variable(vtype='continuous')
    model.add_constraint(aaa == Shape[0] * Shape[1])
    model.add_constraint(bbb == Shape[0] * Shape[2])
    model.add_constraint(ccc == Shape[1] * Shape[2])
    
    model.add_constraint(LINK_cost[0] == num_chips_per_copy * Link_BW[0] * link_unit_price)
    model.add_constraint(LINK_cost[1] == num_chips_per_copy * Link_BW[1] * link_unit_price)
    model.add_constraint(LINK_cost[2] == num_chips_per_copy * Link_BW[2] * link_unit_price)
    
    model.add_constraint(SWITCH_cost[0] == 2 * ccc * Link_BW[0] * switch_unit_price)
    model.add_constraint(SWITCH_cost[1] == 2 * bbb * Link_BW[1] * switch_unit_price)
    model.add_constraint(SWITCH_cost[2] == 2 * aaa * Link_BW[2] * switch_unit_price)

elif dse.system.WhichOneof('topology_variant') == 'r_sw_sw': # 3d dgx-1
    ccc = model.add_variable(vtype='continuous')
    model.add_constraint(ccc == Shape[1] * Shape[2])
    
    model.add_constraint(LINK_cost[0] == num_chips_per_copy * Link_BW[0] * link_unit_price)
    model.add_constraint(LINK_cost[1] == ccc * Link_BW[1] * link_unit_price)
    model.add_constraint(LINK_cost[2] == Shape[2] * Link_BW[2] * link_unit_price)
    
    model.add_constraint(SWITCH_cost[0] == 2 * ccc * Link_BW[0] * switch_unit_price)
    model.add_constraint(SWITCH_cost[1] == ccc * Link_BW[1] * switch_unit_price)
    model.add_constraint(SWITCH_cost[2] == Shape[2] * Link_BW[2] * switch_unit_price)
    
elif dse.system.WhichOneof('topology_variant') == 'sw_sw_sw': # 3d dgx-2
    ccc = model.add_variable(vtype='continuous')
    model.add_constraint(ccc == Shape[1] * Shape[2])
    
    model.add_constraint(LINK_cost[0] == num_chips_per_copy * Link_BW[0] * link_unit_price)
    model.add_constraint(LINK_cost[1] == ccc * Link_BW[1] * link_unit_price)
    model.add_constraint(LINK_cost[2] == Shape[2] * Link_BW[2] * link_unit_price)
    
    model.add_constraint(SWITCH_cost[0] == num_chips_per_copy * Link_BW[0] * switch_unit_price)
    model.add_constraint(SWITCH_cost[1] == ccc * Link_BW[1] * switch_unit_price)
    model.add_constraint(SWITCH_cost[2] == Shape[2] * Link_BW[2] * switch_unit_price)

elif dse.system.WhichOneof('topology_variant') == 'r_r_r_r': # 4d torus
    aaa = model.add_variable(vtype='continuous')
    bbb = model.add_variable(vtype='continuous')
    ccc = model.add_variable(vtype='continuous')
    model.add_constraint(aaa == Shape[0] * Shape[1])
    model.add_constraint(bbb == Shape[0] * Shape[2])
    model.add_constraint(ccc == Shape[1] * Shape[2])
    
    model.add_constraint(LINK_cost[0] == num_chips_per_copy * Link_BW[0] * link_unit_price)
    model.add_constraint(LINK_cost[1] == num_chips_per_copy * Link_BW[1] * link_unit_price)
    model.add_constraint(LINK_cost[2] == num_chips_per_copy * Link_BW[2] * link_unit_price)
    
    model.add_constraint(SWITCH_cost[0] == 2 * ccc * Link_BW[0] * switch_unit_price)
    model.add_constraint(SWITCH_cost[1] == 2 * bbb * Link_BW[1] * switch_unit_price)
    model.add_constraint(SWITCH_cost[2] == 2 * aaa * Link_BW[2] * switch_unit_price)

else:
    raise Exception('Wrong!')      



less_or_equal_one_chip = model.add_variable(name='less_or_equal_one_chip', vtype='binary')
less_or_equal_four_chip = model.add_variable(name='less_or_equal_four_chip', vtype='binary')
model.add_constraint((less_or_equal_one_chip == 1) >> (num_chips_per_copy == 1))
model.add_constraint((less_or_equal_one_chip == 0) >> (num_chips_per_copy >= 2))
model.add_constraint((less_or_equal_four_chip == 1) >> (num_chips_per_copy <= 4))
model.add_constraint((less_or_equal_four_chip == 0) >> (num_chips_per_copy >= 5))



total_DRAM_cost = model.add_variable(name='total_DRAM_cost', vtype='continuous', lb=0)
total_accelerator_cost = model.add_variable(name='total_accelerator_cost', vtype='continuous', lb=0)
total_link_cost = model.add_variable(name='total_link_cost', vtype='continuous', lb=0)
total_switch_cost = model.add_variable(name='total_switch_cost', vtype='continuous', lb=0)
model.add_constraint(total_DRAM_cost == num_chip * DRAM_BW * dram_unit_price)
model.add_constraint(total_accelerator_cost == accelerator_price * num_chip)


xxxx = model.add_variable(vtype='continuous')
yyyy = model.add_variable(vtype='continuous')
model.add_constraint(xxxx == (np.ones((len(topology))) @ LINK_cost) * num_copy)
model.add_constraint(yyyy == (np.ones((len(topology))) @ SWITCH_cost) * num_copy)

model.add_constraint((less_or_equal_one_chip == 0) >> (total_link_cost == xxxx))
model.add_constraint((less_or_equal_one_chip == 1) >> (total_link_cost == 0))
model.add_constraint((less_or_equal_four_chip == 0) >> (total_switch_cost == yyyy))
model.add_constraint((less_or_equal_four_chip == 1) >> (total_switch_cost == 0))



total_cost = model.add_variable(name='total_cost', vtype='continuous', lb=0)
model.add_constraint(total_cost == total_DRAM_cost + total_accelerator_cost + total_link_cost + total_switch_cost)






# power
LINK_power = model.add_matrix_variable(len(topology), name='LINK_power', vtype='continuous', lb=0)
SWITCH_power = model.add_matrix_variable(len(topology), name='SWITCH_power', vtype='continuous', lb=0)

if dse.system.WhichOneof('topology_variant') == 'sw': # switch
    model.add_constraint(LINK_power[0] == Shape[0] * Link_BW[0] * link_unit_power_x)
    model.add_constraint(SWITCH_power[0] == Shape[0] * Link_BW[0] * switch_unit_power)
    
elif dse.system.WhichOneof('topology_variant') == 'fc': # fc
    aaa = model.add_variable(vtype='continuous')
    model.add_constraint(aaa == Shape[0] * (Shape[0] - 1))
    
    model.add_constraint(LINK_power[0] == aaa * Link_BW[0] * link_unit_power_x)
    model.add_constraint(SWITCH_power[0] == 0)
    
elif dse.system.WhichOneof('topology_variant') == 'r': # ring
    model.add_constraint(LINK_power[0] == Shape[0] * Link_BW[0] * link_unit_power_x)
    model.add_constraint(SWITCH_power[0] == 2 * Link_BW[0] * switch_unit_power)
    
elif dse.system.WhichOneof('topology_variant') == 'r_r': # 2d torus
    model.add_constraint(LINK_power[0] == num_chips_per_copy * Link_BW[0] * link_unit_power_x)
    model.add_constraint(LINK_power[1] == num_chips_per_copy * Link_BW[1] * link_unit_power_y)
    
    model.add_constraint(SWITCH_power[0] == 2 * Shape[1] * Link_BW[0] * switch_unit_power)
    model.add_constraint(SWITCH_power[1] == 2 * Shape[0] * Link_BW[1] * switch_unit_power)
    
elif dse.system.WhichOneof('topology_variant') == 'fc_fc': # 2d dragonfly
    aaa = model.add_variable(vtype='continuous')
    bbb = model.add_variable(vtype='continuous')
    model.add_constraint(aaa == Shape[0] * (Shape[0] - 1))
    model.add_constraint(bbb == Shape[1] * (Shape[1] - 1))
    
    ccc = model.add_variable(vtype='continuous')
    model.add_constraint(ccc == aaa * Shape[1])
    model.add_constraint(LINK_power[0] == ccc * Link_BW[0] * link_unit_power_x)
    model.add_constraint(LINK_power[1] == bbb * Link_BW[1] * link_unit_power_y)
    
    ddd = model.add_variable(vtype='continuous')
    model.add_constraint(ddd == Shape[1] * Shape[1])
    model.add_constraint(SWITCH_power[0] == 0)
    model.add_constraint(SWITCH_power[1] == ddd * Link_BW[1] * switch_unit_power)
    
elif dse.system.WhichOneof('topology_variant') == 'r_fc': # zionex
    bbb = model.add_variable(vtype='continuous')
    model.add_constraint(bbb == Shape[1] * (Shape[1] - 1))
    
    model.add_constraint(LINK_power[0] == num_chips_per_copy * Link_BW[0] * link_unit_power_x)
    model.add_constraint(LINK_power[1] == bbb * Link_BW[1] * link_unit_power_y)
    
    ddd = model.add_variable(vtype='continuous')
    model.add_constraint(ddd == Shape[1] * Shape[1])
    model.add_constraint(SWITCH_power[0] == 2 * Shape[1] * Link_BW[0] * switch_unit_power)
    model.add_constraint(SWITCH_power[1] == ddd * Link_BW[1] * switch_unit_power)

elif dse.system.WhichOneof('topology_variant') == 'r_sw': # 2d dgx-1
        model.add_constraint(LINK_power[0] == num_chip * Link_BW[0] * link_unit_power_x)
        model.add_constraint(LINK_power[1] == Shape[1] * Link_BW[1] * link_unit_power_y)
        
        model.add_constraint(SWITCH_power[0] == 2 * Shape[1] * Link_BW[0] * switch_unit_power)
        model.add_constraint(SWITCH_power[1] == Shape[1] * Link_BW[1] * switch_unit_power)    
        
elif dse.system.WhichOneof('topology_variant') == 'sw_sw': # 2d dgx-2
    model.add_constraint(LINK_power[0] == num_chip * Link_BW[0] * link_unit_power_x)
    model.add_constraint(LINK_power[1] == Shape[1] * Link_BW[1] * link_unit_power_y)
    
    model.add_constraint(SWITCH_power[0] == num_chip * Link_BW[0] * switch_unit_power)
    model.add_constraint(SWITCH_power[1] == Shape[1] * Link_BW[1] * switch_unit_power) 

elif dse.system.WhichOneof('topology_variant') == 'r_r_r': # 3d torus
    aaa = model.add_variable(vtype='continuous')
    bbb = model.add_variable(vtype='continuous')
    ccc = model.add_variable(vtype='continuous')
    model.add_constraint(aaa == Shape[0] * Shape[1])
    model.add_constraint(bbb == Shape[0] * Shape[2])
    model.add_constraint(ccc == Shape[1] * Shape[2])
    
    model.add_constraint(LINK_power[0] == num_chips_per_copy * Link_BW[0] * link_unit_power_x)
    model.add_constraint(LINK_power[1] == num_chips_per_copy * Link_BW[1] * link_unit_power_y)
    model.add_constraint(LINK_power[2] == num_chips_per_copy * Link_BW[2] * link_unit_power_z)
    
    model.add_constraint(SWITCH_power[0] == 2 * ccc * Link_BW[0] * switch_unit_power)
    model.add_constraint(SWITCH_power[1] == 2 * bbb * Link_BW[1] * switch_unit_power)
    model.add_constraint(SWITCH_power[2] == 2 * aaa * Link_BW[2] * switch_unit_power)

elif dse.system.WhichOneof('topology_variant') == 'r_sw_sw': # 3d dgx-1
    ccc = model.add_variable(vtype='continuous')
    model.add_constraint(ccc == Shape[1] * Shape[2])
    
    model.add_constraint(LINK_power[0] == num_chips_per_copy * Link_BW[0] * link_unit_power_x)
    model.add_constraint(LINK_power[1] == ccc * Link_BW[1] * link_unit_power_y)
    model.add_constraint(LINK_power[2] == Shape[2] * Link_BW[2] * link_unit_power_z)
    
    model.add_constraint(SWITCH_power[0] == 2 * ccc * Link_BW[0] * switch_unit_power)
    model.add_constraint(SWITCH_power[1] == ccc * Link_BW[1] * switch_unit_power)
    model.add_constraint(SWITCH_power[2] == Shape[2] * Link_BW[2] * switch_unit_power)
    
elif dse.system.WhichOneof('topology_variant') == 'sw_sw_sw': # 3d dgx-2
    ccc = model.add_variable(vtype='continuous')
    model.add_constraint(ccc == Shape[1] * Shape[2])
    
    model.add_constraint(LINK_power[0] == num_chips_per_copy * Link_BW[0] * link_unit_power_x)
    model.add_constraint(LINK_power[1] == ccc * Link_BW[1] * link_unit_power_y)
    model.add_constraint(LINK_power[2] == Shape[2] * Link_BW[2] * link_unit_power_z)
    
    model.add_constraint(SWITCH_power[0] == num_chips_per_copy * Link_BW[0] * switch_unit_power)
    model.add_constraint(SWITCH_power[1] == ccc * Link_BW[1] * switch_unit_power)
    model.add_constraint(SWITCH_power[2] == Shape[2] * Link_BW[2] * switch_unit_power)
       
elif dse.system.WhichOneof('topology_variant') == 'r_r_r_r': # 4d torus
    aaa = model.add_variable(vtype='continuous')
    bbb = model.add_variable(vtype='continuous')
    ccc = model.add_variable(vtype='continuous')
    model.add_constraint(aaa == Shape[0] * Shape[1])
    model.add_constraint(bbb == Shape[0] * Shape[2])
    model.add_constraint(ccc == Shape[1] * Shape[2])
    
    model.add_constraint(LINK_power[0] == num_chips_per_copy * Link_BW[0] * link_unit_power_x)
    model.add_constraint(LINK_power[1] == num_chips_per_copy * Link_BW[1] * link_unit_power_y)
    model.add_constraint(LINK_power[2] == num_chips_per_copy * Link_BW[2] * link_unit_power_z)
    
    model.add_constraint(SWITCH_power[0] == 2 * ccc * Link_BW[0] * switch_unit_power)
    model.add_constraint(SWITCH_power[1] == 2 * bbb * Link_BW[1] * switch_unit_power)
    model.add_constraint(SWITCH_power[2] == 2 * aaa * Link_BW[2] * switch_unit_power)

else:
    raise Exception('Wrong!')


total_DRAM_power = model.add_variable(name='total_DRAM_power', vtype='continuous', lb=0)
total_accelerator_power = model.add_variable(name='total_accelerator_power', vtype='continuous', lb=0)
total_link_power = model.add_variable(name='total_link_power', vtype='continuous', lb=0)
total_switch_power = model.add_variable(name='total_switch_power', vtype='continuous', lb=0)

model.add_constraint(total_DRAM_power == num_chip * DRAM_BW * dram_unit_power)
model.add_constraint(total_accelerator_power == accelerator_power * num_chip)



aaaa = model.add_variable(vtype='continuous')
bbbb = model.add_variable(vtype='continuous')
model.add_constraint(aaaa == (np.ones((len(topology))) @ LINK_power) * num_copy)
model.add_constraint(bbbb == (np.ones((len(topology))) @ SWITCH_power) * num_copy)

model.add_constraint((less_or_equal_one_chip == 0) >> (total_link_power == aaaa))
model.add_constraint((less_or_equal_one_chip == 1) >> (total_link_power == 0))
model.add_constraint((less_or_equal_four_chip == 0) >> (total_switch_power == bbbb))
model.add_constraint((less_or_equal_four_chip == 1) >> (total_switch_power == 0))



total_power = model.add_variable(name='total_power', vtype='continuous', lb=0)
model.add_constraint(total_power == total_DRAM_power + total_accelerator_power + total_link_power + total_switch_power)




# objective
final_ii_ns = model.add_variable(name='final_ii_ns', vtype='continuous', lb=0)
model.add_constraint(final_ii_ns * num_copy >= ns_per_batch)

total_compute_latency = model.add_variable(name='total_compute_latency', vtype='continuous', lb=0)
model.add_constraint(total_compute_latency == np.ones((C)) @ Compute_Latency)

total_memory_latency = model.add_variable(name='total_memory_latency', vtype='continuous', lb=0)
model.add_constraint(total_memory_latency == np.ones((C)) @ Memory_Latency)

total_network_latency = model.add_variable(name='total_network_latency', vtype='continuous', lb=0)
model.add_constraint(total_network_latency == np.ones((C)) @ Network_Latency)

if p_and_r_flag:
    total_pcu_pmu_used = model.add_variable(name='total_pcu_pmu_used', vtype='continuous', lb=0)
    model.add_constraint(total_pcu_pmu_used == PMU_used_per_config @ np.ones((C)) + Par_total @ np.ones((num_node)))

    model.set_objective_minimize(final_ii_ns)  # Use HiGHS’s objective setter
    model.set_objectiveN(total_pcu_pmu_used, index=1, priority=3)
    model.set_objectiveN(total_compute_latency, index=2, priority=2)
    model.set_objectiveN(total_memory_latency, index=3, priority=1)
    model.set_objectiveN(total_network_latency, index=4, priority=0)
    model.ModelSense = highspy.MINIMIZE

else:
    if first_bwd_kernel == -1 and dse.execution.WhichOneof('workload_variant') == 'llm': # inference
        model.set_objective_minimize(final_ii_ns)  # Use HiGHS’s objective setter
        model.set_objectiveN(final_latency_ns, index=1, priority=3)
        model.set_objectiveN(total_compute_latency, index=2, priority=2)
        model.set_objectiveN(total_memory_latency, index=3, priority=1)
        model.set_objectiveN(total_network_latency, index=4, priority=0)
        model.ModelSense = highspy.MINIMIZE
    else:
        model.set_objective_minimize(final_ii_ns)  # Use HiGHS’s objective setter
        model.set_objectiveN(total_compute_latency, index=1, priority=2)
        model.set_objectiveN(total_memory_latency, index=2, priority=1)
        model.set_objectiveN(total_network_latency, index=3, priority=0)
        model.ModelSense = highspy.MINIMIZE
model.solve()


end = time.time()





# get values from gurobi
shard_M = []
shard_K = []
shard_N = []
shard_intermediate_buffer_size = []
shard_initiation_buffer_size = []
Config = []
Link_BW = []
Per_Config_II = []


Memory_Latency = []
Compute_Latency = []
Network_Latency = []
A = []
Z = []
Par_lane = []
Par_stage = []
PMU_used_intermediate = []
PMU_used_initiation = []
Cycle = []


MMM = []
KKK = []
NNN = []
layers = 0
num_tile = 0
BROADCAST_communication_size = []
POINT_TO_POINT_communication_size = []
ALL_TO_ALL_communication_size_node = []
ALL_REDUCE_communication_size_node = []
num_tile_per_config = []

for v in model.get_variables():
    print(v.name, v.x)

    if v.name.startswith('num_tile'):
        num_tile = v.x

    if v.name.startswith('A['):
        A.append(v.x)
    if v.name.startswith('Z['):
        Z.append(v.x)
    if v.name.startswith('PMU_used_intermediate['):
        PMU_used_intermediate.append(v.x)
    if v.name.startswith('PMU_used_initiation['):
        PMU_used_initiation.append(v.x)
    if v.name.startswith('Cycle['):
        Cycle.append(v.x)
    
    if v.name.startswith('MMM['):
        MMM.append(v.x)
    if v.name.startswith('KKK['):
        KKK.append(v.x)
    if v.name.startswith('NNN['):
        NNN.append(v.x)
        
    if v.name.startswith('Par_lane['):
        Par_lane.append(v.x)
    if v.name.startswith('Par_stage['):
        Par_stage.append(v.x)    
        
    if v.name.startswith('Memory_Latency['):
        Memory_Latency.append(v.x)
    if v.name.startswith('Compute_Latency['):
        Compute_Latency.append(v.x)
    if v.name.startswith('Network_Latency['):
        Network_Latency.append(v.x)
    
    
    
    
    if v.name.startswith('shard_M'):
        shard_M.append(v.x)
    if v.name.startswith('shard_K'):
        shard_K.append(v.x)
    if v.name.startswith('shard_N'):
        shard_N.append(v.x)
    if v.name.startswith('shard_intermediate_buffer_size'):
        shard_intermediate_buffer_size.append(v.x)
    if v.name.startswith('shard_initiation_buffer_size'):
        shard_initiation_buffer_size.append(v.x)
    if v.name == 'final_ii_ns':
        final_ii_ns = v.x
    # if v.name.startswith('total_DRAM_bytes'):
        # total_DRAM_bytes = v.x
    # if v.name.startswith('total_Network_bytes'):
        # total_Network_bytes = v.x
    if v.name.startswith('TP'):
        TP = v.x
    if v.name.startswith('PP'):
        PP = v.x
    if v.name.startswith('DP'):
        DP = v.x
    if v.name.startswith('Config'):
        Config.append(v.x)
    if v.name.startswith('total_cost'):
        total_cost = v.x
    if v.name.startswith('total_power'):
        total_power = v.x 
    if v.name.startswith('DRAM_BW'):
        DRAM_BW = v.x 
    if v.name.startswith('Micro_Batch_Size'):
        Micro_Batch_Size = v.x 
    if v.name.startswith('Link_BW['):
        Link_BW.append(v.x) 
    if v.name.startswith('Per_Config_II'):
        Per_Config_II.append(v.x)
    if v.name.startswith('num_micro_batch_per_pipeline'):
        num_micro_batch_per_pipeline = v.x

    if v.name.startswith('layers'):
        layers = v.x

    if v.name.startswith('BROADCAST_communication_size'):
        BROADCAST_communication_size.append(v.x)
    if v.name.startswith('POINT_TO_POINT_communication_size'):
        POINT_TO_POINT_communication_size.append(v.x)
    if v.name.startswith('ALL_TO_ALL_communication_size_node'):
        ALL_TO_ALL_communication_size_node.append(v.x)
    if v.name.startswith('ALL_REDUCE_communication_size_node'):
        ALL_REDUCE_communication_size_node.append(v.x)


    if v.name.startswith('num_tile_per_config'):
        num_tile_per_config.append(v.x)

# log latencies to hdf5 file
f = h5py.File(name+'/'+'log.hdf5', 'w')
ds1 = f.create_dataset('Memory_Latency', len(Memory_Latency))
for i in range(len(Memory_Latency)):
    ds1[i] = Memory_Latency[i]

ds2 = f.create_dataset('Compute_Latency', len(Memory_Latency))
for i in range(len(Compute_Latency)):
    ds2[i] = Compute_Latency[i]

ds3 = f.create_dataset('Network_Latency', len(Memory_Latency))   
for i in range(len(Network_Latency)):
    ds3[i] = Network_Latency[i]
    
f.close()




print('------------Statistics------------')
print('FLOP per kernel:')      

FLOP = 0.0
for i in range(len(M)):
    if kernel_type[i] == KernelType.SIMD.value:
        tmp = M[i] * K[i] * N[i] * num_input[i]
        FLOP += tmp
        print('SIMD', kernel_name[i], M[i], K[i], N[i], num_input[i], tmp)

    elif kernel_type[i] == KernelType.SYSTOLIC.value:
        tmp = 2 * M[i] * K[i] * N[i] * num_input[i]
        FLOP += tmp
        print('SYSTOLIC', kernel_name[i], M[i], K[i], N[i], num_input[i], tmp)

for element in memory_size:
    print('memory_size', element)


print('kernel_name', kernel_name)

for i in range(num_edge):
    print(kernel_name[node_dict[startIdx[i]]], kernel_name[node_dict[endIdx[i]]])

print('depth')
for i in range(num_edge):
    print(depth[i])



FLOP *= layers * global_batch_size
final_ii_s = final_ii_ns/1e9
util = FLOP/final_ii_s/num_chip/GFLOPS/1e9
    

print()
print()
print()

if len(dimension) == 4:
    print('SP', dimension[0])
print('TP', TP)   
print('PP', PP)   
print('DP', DP)
print('final_ii_s', final_ii_s)
print('Number of Chips', num_chip)
print('Per-Accelerator Throughput (GFLOPS)', GFLOPS)
print('DRAM BW', DRAM_BW)
print('Link BW', Link_BW)
print('System Cost', total_cost)
print('System Power', total_power)
print('Workload FLOP', FLOP)
print('System FLOPS Utilization', util)
print('Optimizer Runtime (s)', end - start)



# update kernels
i = 0
for kernel in dse.dataflow_graph.kernels:
    if kernel.WhichOneof('kernel_variant') == 'gemm_input1_weight':
        kernel.gemm_input1_weight.shard_outer_M = int(shard_M[i])
        kernel.gemm_input1_weight.shard_K = int(shard_K[i])
        kernel.gemm_input1_weight.shard_N = int(shard_N[i])
        kernel.config = int(Config[i])
    
    elif kernel.WhichOneof('kernel_variant') == 'gemm_input1_input2':
        kernel.gemm_input1_input2.shard_outer_M = int(shard_M[i])
        kernel.gemm_input1_input2.shard_K = int(shard_N[i])
        kernel.gemm_input1_input2.shard_N = int(shard_N[i])
        kernel.config = int(Config[i])
        
    elif kernel.WhichOneof('kernel_variant') == 'elementwise_input1':
        kernel.elementwise_input1.shard_outer_M = int(shard_M[i])
        kernel.elementwise_input1.shard_N = int(shard_N[i])
        kernel.config = int(Config[i])
        
    elif kernel.WhichOneof('kernel_variant') == 'elementwise_input1_input2':
        kernel.elementwise_input1_input2.shard_outer_M = int(shard_M[i])
        kernel.elementwise_input1_input2.shard_N = int(shard_N[i])
        kernel.config = int(Config[i])
        
    else:
        raise Exception('Wrong!')
   
    i += 1




# update edges
i = 0
for connection in dse.dataflow_graph.connections:
    connection.shard_tensor_size = float(shard_intermediate_buffer_size[i])
    
    if p_and_r_flag == True:
        if lane_stage_type[i] == 'lane':
            connection.lane_stage_type = 1
        elif lane_stage_type[i] == 'stage':
            connection.lane_stage_type = 2
        else:
            raise Exception('Wrong!')
    i += 1





# write to final binary
with open('./'+name+'/'+'dse_final.pb', "wb") as file:
    file.write(dse.SerializeToString())


# write to final text file
with open('./'+name+'/'+'dse_final.txt', "w") as file:
    text_format.PrintMessage(dse, file)




# create dot graph
node_list = []
edge_list = []
dict = {}
graph = pydot.Dot(graph_type='digraph')

cluster_dict = {}
for kernel in dse.dataflow_graph.kernels:
    label = text_format.MessageToString(kernel)
    pydot_node = pydot.Node(kernel.name, fillcolor="white", style="filled", label=label, penwidth=8)
    if kernel.config in cluster_dict.keys():
        cluster_dict[kernel.config].append(pydot_node)
    else:
        cluster_dict[kernel.config] = [pydot_node]
    dict[kernel.id] = pydot_node
    graph.add_node(pydot_node)

for connection in dse.dataflow_graph.connections:
    label = text_format.MessageToString(connection)
    pydot_edge = pydot.Edge(dict[connection.startIdx], dict[connection.endIdx], label=label, penwidth=8)
    graph.add_edge(pydot_edge)

for config in cluster_dict.keys():
    cluster = pydot.Cluster('Config: '+str(config), label='Config: '+str(config), color='red', penwidth=8)
    for node in cluster_dict[config]:
        cluster.add_node(node)
    graph.add_subgraph(cluster)


graph.write_png('./'+name+'/'+'dataflow_graph_final.png')

print('\n\n\n')





if not dse.execution.turn_off_p_and_r:
    class pcu():
        def __init__(self):
            self.x = -1
            self.y = -1
            self.cycle = -1
            self.kernel_idx = -1
            self.name = 'N/A'
            self.M = -1
            self.K = -1
            self.N = -1
            self.SIMD_or_Systolic = 'N/A'
        
        def set_pcu_x(self, x):
            self.x = x

        def set_pcu_y(self, y):
            self.y = y

        def set_pcu_cycle(self, cycle):
            self.cycle = cycle
            
        def set_pcu_kernel_idx(self, kernel_idx):    
            self.kernel_idx = kernel_idx
            
        def set_pcu_name(self, name):
            self.name = name
        
        def set_pcu_M(self, M):
            self.M = M 
        
        def set_pcu_K(self, K):
            self.K = K 
            
        def set_pcu_N(self, N):
            self.N = N 
            
        def set_pcu_SIMD_or_Systolic(self, SIMD_or_Systolic):
            self.SIMD_or_Systolic = SIMD_or_Systolic
            
    class pmu():
        def __init__(self):
            self.x = -1
            self.y = -1
            self.upstream_cycle = -1
            self.downstream_cycle = -1
            self.tensor_idx = -1
            self.name = 'N/A'
        
        def set_pmu_x(self, x):
            self.x = x

        def set_pmu_y(self, y):
            self.y = y

        def set_pmu_upstream(self, upstream_cycle):
            self.upstream_cycle = upstream_cycle
            
        def set_pmu_downstream(self, downstream_cycle):
            self.downstream_cycle = downstream_cycle     
        
        def set_pmu_tensor_idx(self, tensor_idx):
            self.tensor_idx = tensor_idx  
        
        def set_pmu_name(self, name):
            self.name = name  
            

    if p_and_r_flag == True:
        
        kernel_to_name = {}
        
        config_to_kernel = {}
        for i in range(C):
            config_to_kernel[i] = []
            
        for i in range(num_node):
            for j in range(C):
                if A[i * C + j] == 1:
                    config_to_kernel[j].append(i)
                    
                    
                    
        kernel_to_num_pcu = {}
        kernel_to_cycle = {}
        kernel_to_M = {}
        kernel_to_K = {}
        kernel_to_N = {}
        kernel_to_SIMD_Systolic = {}
        for i in range(num_node):
            kernel_to_num_pcu[i] = round(Par_lane[i] * Par_stage[i])
            kernel_to_cycle[i] = round(Cycle[i])
            kernel_to_M[i] = round(MMM[i])
            kernel_to_K[i] = round(KKK[i])
            kernel_to_N[i] = round(NNN[i])
            kernel_to_SIMD_Systolic[i] = SIMD_or_Systolic[i]
        
        kernel_to_upstream_edge = {}
        kernel_to_downstream_edge = {}
        kernel_to_weight = {}
        
        for i in range(num_node):
            kernel_to_upstream_edge[i] = []
            kernel_to_downstream_edge[i] = []
            
        
        for i in range(num_node):
            if i in upstream_edge_dict.keys():
                for j in upstream_edge_dict[i]:
                    kernel_to_upstream_edge[i].append((j, round(PMU_used_intermediate[j])))    
        for i in range(num_node):
            if i in downstream_edge_dict.keys():
                for j in downstream_edge_dict[i]:
                    kernel_to_downstream_edge[i].append((j, round(PMU_used_intermediate[j])))           
        for i in range(num_weight):
            kernel_to_weight[node_dict[weight_dict[i]]] = (i+num_edge, round(PMU_used_initiation[i]))

        
        if dse.system.accelerator.x * dse.system.accelerator.y != dse.system.accelerator.core + dse.system.accelerator.pmu:
            raise Exception('Wrong!')
        else:
            X = dse.system.accelerator.x
            Y = dse.system.accelerator.y

            hardware_pcu = []
            hardware_pmu = []

            if dse.system.accelerator.placement == "rowwise":
                for x in range(X):
                    for y in range(Y):
                        if x % 2 == 0:
                            if y % 2 == 1:
                                hardware_pcu.append((x, y))
                        else:
                            if y % 2 == 0:
                                hardware_pcu.append((x, y))
                
                for x in range(X):
                    for y in range(Y):
                        if x % 2 == 0:
                            if y % 2 == 0:
                                hardware_pmu.append((x, y))
                        else:
                            if y % 2 == 1:
                                hardware_pmu.append((x, y))

            elif dse.system.accelerator.placement == "diagonalwise":
                tmp = X+Y
                for i in range(tmp):
                    for j in range(i+1):
                        y = j
                        x = i - j
                        if 0 <= x < X and 0 <= y < Y:
                            if x % 2 == 0:
                                if y % 2 == 1:
                                    hardware_pcu.append((x, y))
                            else:
                                if y % 2 == 0:
                                    hardware_pcu.append((x, y))
                
                for i in range(tmp):
                    for j in range(i+1):
                        y = j
                        x = i - j
                        if 0 <= x < X and 0 <= y < Y:
                            if x % 2 == 0:
                                if y % 2 == 0:
                                    hardware_pmu.append((x, y))
                            else:
                                if y % 2 == 1:
                                    hardware_pmu.append((x, y))

            else:
                raise Exception("Wrong!")









            
            for i in range(C):
                print('------------------- Config', i, '-----------------------')

                pcu_arr = []
                pmu_arr = []
                        
                tensor_to_upstream_cycle = {}
                tensor_to_downstream_cycle = {}
                tensor_to_upstream_name = {}
                tensor_to_downstream_name = {}
                tensor_to_pmu_used = {}
                tensor_to_pmu_placement = {}
                tensor_to_pcu_placement = {}


                
                
                
                        
                cnt = 0      
                for j in config_to_kernel[i]:
                    kernel_idx = j
                    num_pcu = kernel_to_num_pcu[kernel_idx]
                    cycle = kernel_to_cycle[kernel_idx]
                    name = kernel_name[kernel_idx]
                    M = kernel_to_M[kernel_idx]
                    K = kernel_to_K[kernel_idx]
                    N = kernel_to_N[kernel_idx]
                    SIMD_or_Systolic = kernel_to_SIMD_Systolic[kernel_idx]
                    
                    tensor_to_pcu_placement[kernel_idx] = (cnt, cnt+num_pcu-1)
                    
                    for _ in range(num_pcu):
                        my_pcu = pcu()
                        my_pcu.set_pcu_cycle(cycle)
                        my_pcu.set_pcu_kernel_idx(kernel_idx)
                        my_pcu.set_pcu_name(name)
                        my_pcu.set_pcu_M(M)
                        my_pcu.set_pcu_K(K)
                        my_pcu.set_pcu_N(N)
                        my_pcu.set_pcu_SIMD_or_Systolic(SIMD_or_Systolic)
                        pcu_arr.append(my_pcu)
                        cnt += 1

                
                cnt = 0
                for j in config_to_kernel[i]:
                    kernel_idx = j
                    cycle = kernel_to_cycle[kernel_idx]
                    name = kernel_name[kernel_idx]
                    
                    for tensor_idx, pmu_used in kernel_to_upstream_edge[kernel_idx]:
                        tensor_to_pmu_used[tensor_idx] = pmu_used
                        tensor_to_downstream_cycle[tensor_idx] = cycle
                        tensor_to_downstream_name[tensor_idx] = name
                        
                    for tensor_idx, pmu_used in kernel_to_downstream_edge[kernel_idx]:
                        tensor_to_pmu_used[tensor_idx] = pmu_used
                        tensor_to_upstream_cycle[tensor_idx] = cycle
                        tensor_to_upstream_name[tensor_idx] = name
                    
                    if kernel_idx in kernel_to_weight.keys():
                        tensor_idx = kernel_to_weight[kernel_idx][0]
                        pmu_used = kernel_to_weight[kernel_idx][1]
                        tensor_to_pmu_used[tensor_idx] = pmu_used
                        tensor_to_downstream_cycle[tensor_idx] = cycle
                        tensor_to_downstream_name[tensor_idx] = name



                cnt = 0  
                for j in tensor_to_pmu_used.keys():
                    tensor_idx = j
                    pmu_used = tensor_to_pmu_used[j]
                    
                    if tensor_idx not in tensor_to_upstream_cycle.keys():
                        upstream_cycle = 0
                        upstream_name = 'N/A'
                    else:
                        upstream_cycle = tensor_to_upstream_cycle[tensor_idx]
                        upstream_name = tensor_to_upstream_name[tensor_idx]
                        
                    if tensor_idx not in tensor_to_downstream_cycle.keys():
                        downstream_cycle = 0
                        downstream_name = 'N/A'
                    else:
                        downstream_cycle = tensor_to_downstream_cycle[tensor_idx]
                        downstream_name = tensor_to_downstream_name[tensor_idx]
                    
                    tensor_to_pmu_placement[tensor_idx] = (cnt, cnt+pmu_used-1)
                    
                    for _ in range(pmu_used):
                        my_pmu = pmu()
                        my_pmu.set_pmu_upstream(upstream_cycle)
                        my_pmu.set_pmu_downstream(downstream_cycle)
                        my_pmu.set_pmu_tensor_idx(tensor_idx)
                        my_pmu.set_pmu_name(upstream_name+'___'+downstream_name)
                        pmu_arr.append(my_pmu)
                        cnt += 1
                
                
                
                
                
                connection_list = []
                connection_list_sender = []
                connection_list_receiver = []
                
                for j in config_to_kernel[i]:
                    kernel_idx = j
                    kernel_start = tensor_to_pcu_placement[kernel_idx][0]
                    kernel_end = tensor_to_pcu_placement[kernel_idx][1]
                    num_pcu = kernel_end - kernel_start + 1
                    
                    for tensor_idx, _ in kernel_to_upstream_edge[kernel_idx]: # tensor to kernel
                        tensor_start = tensor_to_pmu_placement[tensor_idx][0]
                        tensor_end = tensor_to_pmu_placement[tensor_idx][1]
                        num_pmu = tensor_end - tensor_start + 1

                        if num_pmu >= num_pcu:
                            ratio = math.floor(num_pmu / num_pcu)
                            for k in range(num_pcu):
                                for l in range(ratio):
                                    tmp = ('pmu', tensor_start+k*ratio+l, 'pcu', kernel_start+k)
                                    connection_list.append(tmp)
                                    
                            for k in range(tensor_start+num_pcu*ratio, tensor_end+1):
                                tmp = ('pmu', k, 'pcu', kernel_end)
                                connection_list.append(tmp)

                        else:
                            ratio = math.floor(num_pcu / num_pmu)
                            for k in range(num_pmu):
                                for l in range(ratio):
                                    tmp = ('pmu', tensor_start+k, 'pcu', kernel_start+k*ratio+l)
                                    connection_list.append(tmp)
                                    
                            for k in range(kernel_start+num_pmu*ratio, kernel_end+1):
                                tmp = ('pmu', tensor_end, 'pcu', k)
                                connection_list.append(tmp)

                    for tensor_idx, _ in kernel_to_downstream_edge[kernel_idx]: # kernel to tensor
                        tensor_start = tensor_to_pmu_placement[tensor_idx][0]
                        tensor_end = tensor_to_pmu_placement[tensor_idx][1]
                        num_pmu = tensor_end - tensor_start + 1

                        if num_pcu >= num_pmu:
                            ratio = math.floor(num_pcu / num_pmu)
                            for k in range(num_pmu):
                                for l in range(ratio):
                                    tmp = ('pcu', kernel_start+k*ratio+l, 'pmu', tensor_start+k)
                                    connection_list.append(tmp)
                                    
                            for k in range(kernel_start+num_pmu*ratio, kernel_end+1):
                                tmp = ('pcu', k, 'pmu', tensor_end)
                                connection_list.append(tmp)

                        else:
                            ratio = math.floor(num_pmu / num_pcu)
                            for k in range(num_pcu):
                                for l in range(ratio):
                                    tmp = ('pcu', kernel_start+k, 'pmu', tensor_start+k*ratio+l)
                                    connection_list.append(tmp)
                                    
                            for k in range(tensor_start+num_pcu*ratio, tensor_end+1):
                                tmp = ('pcu', kernel_end, 'pmu', k)
                                connection_list.append(tmp)



                    if kernel_idx in kernel_to_weight.keys(): # weight tensor to kernel
                        tensor_idx, _ = kernel_to_weight[kernel_idx]
                        tensor_start = tensor_to_pmu_placement[tensor_idx][0]
                        tensor_end = tensor_to_pmu_placement[tensor_idx][1]
                        num_pmu = tensor_end - tensor_start + 1

                        if num_pmu >= num_pcu:
                            ratio = math.floor(num_pmu / num_pcu)
                            for k in range(num_pcu):
                                for l in range(ratio):
                                    tmp = ('pmu', tensor_start+k*ratio+l, 'pcu', kernel_start+k)
                                    connection_list.append(tmp)
                                    
                            for k in range(tensor_start+num_pcu*ratio, tensor_end+1):
                                tmp = ('pmu', k, 'pcu', kernel_end)
                                connection_list.append(tmp)

                        else:
                            ratio = math.floor(num_pcu / num_pmu)
                            for k in range(num_pmu):
                                for l in range(ratio):
                                    tmp = ('pmu', tensor_start+k, 'pcu', kernel_start+k*ratio+l)
                                    connection_list.append(tmp)
                                    
                            for k in range(kernel_start+num_pmu*ratio, kernel_end+1):
                                tmp = ('pmu', tensor_end, 'pcu', k)
                                connection_list.append(tmp)



                for a,b,c,d in connection_list:
                    connection_list_sender.append((a, b))
                    connection_list_receiver.append((c, d))
                
                
                


                # PCU
                pcu_list_to_print = []
                cnt = 0
                for j in range(len(pcu_arr)):
                    if pcu_arr[j].cycle != -1: 
                        sender_list = []
                        for ii in range(len(connection_list_sender)):
                            if ('pcu', j) == connection_list_sender[ii]:
                                sender_list.append(ii)
                        
                        if len(sender_list) == 0:
                            sender_list = [999999]
                        
                        receiver_list = []
                        for ii in range(len(connection_list_receiver)):
                            if ('pcu', j) == connection_list_receiver[ii]:
                                receiver_list.append(ii)
                        
                        if len(receiver_list) == 0:
                            receiver_list = [999999]
                        

                        
                        pcu_arr[j].set_pcu_x(hardware_pcu[cnt][0])
                        pcu_arr[j].set_pcu_y(hardware_pcu[cnt][1])
                        
                        pcu_list_to_print.append((pcu_arr[j].name, pcu_arr[j].x, pcu_arr[j].y, round(pcu_arr[j].cycle), pcu_arr[j].SIMD_or_Systolic, sender_list, receiver_list))

                        cnt += 1









                # PMU
                cnt = 0
                pmu_list_to_print = []
                for j in range(len(pmu_arr)):
                    if pmu_arr[j].upstream_cycle != -1 and pmu_arr[j].downstream_cycle != -1:
                        sender_list = []
                        for ii in range(len(connection_list_sender)):
                            if ('pmu', j) == connection_list_sender[ii]:
                                sender_list.append(ii)
                        
                        if len(sender_list) == 0:
                            sender_list = [999999]
                        
                        receiver_list = []
                        for ii in range(len(connection_list_receiver)):
                            if ('pmu', j) == connection_list_receiver[ii]:
                                receiver_list.append(ii)
                        
                        if len(receiver_list) == 0:
                            receiver_list = [999999]
                        


                        pmu_arr[j].set_pmu_x(hardware_pmu[cnt][0])
                        pmu_arr[j].set_pmu_y(hardware_pmu[cnt][1])

                        pmu_list_to_print.append((pmu_arr[j].name, pmu_arr[j].x, pmu_arr[j].y, round(max(pmu_arr[j].upstream_cycle, pmu_arr[j].downstream_cycle)), sender_list, receiver_list))

                        cnt += 1
                            
                



                connection_list_to_print = []
                for a,b,c,d in connection_list:
                    first = a
                    second = c

                    if first == 'pcu':
                        first_x = pcu_arr[b].x
                        first_y = pcu_arr[b].y
                    elif first == 'pmu':
                        first_x = pmu_arr[b].x
                        first_y = pmu_arr[b].y
                    else:
                        raise Exception("Wrong!")
                    
                    if second == 'pcu':
                        second_x = pcu_arr[d].x
                        second_y = pcu_arr[d].y
                    elif second == 'pmu':
                        second_x = pmu_arr[d].x
                        second_y = pmu_arr[d].y
                    else:
                        raise Exception("Wrong!")
                    
                    connection_list_to_print.append((first, first_x, first_y, second, second_x, second_y))






                # print on-chip traffic
                print('on-chip traffic -----------------------------')
                print('PCU array')
                for aa,bb,cc,dd,ee,ff,gg in pcu_list_to_print:
                    print('pcu on_chip_config',i,aa,bb,cc,dd,ee,'sender',end=' ')
                    for ele in ff:
                        print(ele, end=' ')
                    print('receiver', end=' ')
                    for ele in gg:
                        print(ele, end=' ')
                    print()
                
                print('PMU array')
                for aa,bb,cc,dd,ee,ff in pmu_list_to_print:
                    print('pmu on_chip_config',i,aa,bb,cc,dd,'sender',end=' ')
                    for ele in ee:
                        print(ele, end=' ')
                    print('receiver', end=' ')
                    for ele in ff:
                        print(ele, end=' ')
                    print()

                for aa,bb,cc,dd,ee,ff in connection_list_to_print:
                    print('connection on_chip_config',i,aa,bb,cc,dd,ee,ff)
                print('num_of_connections', 'on_chip_config', i, len(connection_list_to_print))
                







                # off-chip traffic
                connection_list_to_print = []
            
                
                if dse.execution.WhichOneof('workload_variant') == 'llm':
                    if len(dimension) != 2:
                        raise Exception('Wrong!')

                    if dse.system.WhichOneof('topology_variant') == 'r_r':
                        network_bytes = 0
                        for m in range(num_node):
                            if int(Config[m]) == i:
                                network_bytes += ALL_REDUCE_communication_size_node[m] * num_tile_per_config[i]

                        network_bytes = network_bytes * (TP-1) / TP;

                        # all-reduce traffic
                        if network_bytes != 0:
                            for y in range(dimension[1]):
                                for x in range(dimension[0]):
                                    connection_list_to_print.append(('router', x, y, 'router', (x+1)%dimension[0], y))

                        print('network_bytes off_chip_config', i, network_bytes)

                    elif dse.system.WhichOneof('topology_variant') == 'fc_fc':
                        network_bytes = 0
                        for m in range(num_node):
                            if int(Config[m]) == i:
                                network_bytes += ALL_REDUCE_communication_size_node[m] * num_tile_per_config[i]

                        network_bytes = network_bytes * 1 / TP;

                        # all-reduce traffic
                        if network_bytes != 0:
                            for y in range(dimension[1]):
                                for x1 in range(dimension[0]):
                                    for x2 in range(dimension[0]):
                                        if x1 != x2:
                                            connection_list_to_print.append(('router', x1, y, 'router', x2, y))

                        print('network_bytes off_chip_config', i, network_bytes)





                elif dse.execution.WhichOneof('workload_variant') == 'hpl':
                    if len(dimension) != 2:
                        raise Exception('Wrong!')
                    
                    network_bytes = 0
                    for m in range(num_node):
                        if int(Config[m]) == i:
                            network_bytes += BROADCAST_communication_size[m] * num_tile_per_config[i]

                    # broadcast traffic
                    if network_bytes != 0:
                        for x in range(dimension[0]-1):
                            for y in range(dimension[1]):
                                connection_list_to_print.append(('router', x, y, 'router', (x+1)%dimension[0], y))

                    print('network_bytes off_chip_config', i, network_bytes)





                elif dse.execution.WhichOneof('workload_variant') == 'fft' or dse.execution.WhichOneof('workload_variant') == 'dlrm':
                    if len(dimension) != 2:
                        raise Exception('Wrong!')

                    network_bytes = 0
                    for m in range(num_node):
                        if int(Config[m]) == i:
                            network_bytes += ALL_TO_ALL_communication_size_node[m] * num_tile_per_config[i]


                    # all-to-all traffic
                    if network_bytes != 0:
                        for x1 in range(dimension[0]):
                            for y1 in range(dimension[1]):
                                for x2 in range(dimension[0]):
                                    for y2 in range(dimension[1]):
                                        if x1 * dimension[1] + y1 != x2 * dimension[1] + y2:
                                            connection_list_to_print.append(('router', x1, y1, 'router', x2, y2))

                    print('network_bytes off_chip_config', i, network_bytes)
                                            



                # print off-chip traffic
                print('off-chip traffic -----------------------------')
                for aa,bb,cc,dd,ee,ff in connection_list_to_print:
                    print('connection off_chip_config',i,aa,bb,cc,dd,ee,ff)     
                print('num_of_connections', 'off_chip_config', i, len(connection_list_to_print))

                print('\n\n\n\n\n\n\n\n')


