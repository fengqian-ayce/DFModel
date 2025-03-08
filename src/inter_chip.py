from highspy import Highs  # Assumes HiGHS Python API
import argparse
import numpy as np
import setup_pb2
import pprint
from enum import Enum
from google.protobuf import text_format
import pydot
import copy
import os

# user pass in
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help='pls pass in the folder name under the current directory (named after the DL model)', required=True)
args = parser.parse_args()
name = args.name


# read in pd file
dse = setup_pb2.DSE()
with open('./'+name+'/'+'dse.pb', "rb") as file:
    dse.ParseFromString(file.read())
    
    


class Execution_Style(Enum):
    NO_Execution_Style = 0
    DATAFLOW = 1
    KERNEL_BY_KERNEL = 2
    
class KernelType(Enum):
    SYSTOLIC = 1
    SIMD = 2



class Communication(Enum):
    NO_COMMUNICATION = 0
    ALL_REDUCE = 1
    ALL_TO_ALL = 2
    ALL_GATHER = 3
    ALL_REDUCE_PERIODIC = 4


class Dim(Enum):
    OUTER = 0
    M = 1
    K = 2
    N = 3
    NO_DIM = 4
    

class Tensor(Enum):
    RR = 0
    RS = 1
    SR = 2
    
    
class FWD_BWD(Enum):
    Placeholder = 0
    FWD = 1
    BWD = 2
    
class BasicTopology(Enum):
    NO_BASICTOPOLOGY = 0
    R = 1
    FC = 2
    SW = 3


    

# get dataflow graph info
kernel_name = []
output_tensor_size = []
kernel_type = []
outer = []
M = []
K = []
N = []
weight_tensor_size = []
input_tensor_1_size = []
input_tensor_2_size = []
tiling = []
configs = []
fwd_bwd = []
node_dict = {}
i = 0
for kernel in dse.dataflow_graph.kernels:
    kernel_name.append(kernel.name)
    kernel_type.append(kernel.type)
    configs.append(kernel.config)
    fwd_bwd.append(kernel.fwd_bwd)
    
    if kernel.WhichOneof('kernel_variant') == 'gemm_input1_weight':  
        outer.append(kernel.gemm_input1_weight.outer)
        M.append(kernel.gemm_input1_weight.M)
        K.append(kernel.gemm_input1_weight.K)
        N.append(kernel.gemm_input1_weight.N)
        
        input_tensor_1_size.append(kernel.gemm_input1_weight.input_tensor_size)
        input_tensor_2_size.append(-1.0)
        weight_tensor_size.append(kernel.gemm_input1_weight.weight_tensor_size)
        output_tensor_size.append(kernel.gemm_input1_weight.output_tensor_size)
        
        tiling.append(kernel.gemm_input1_weight.tiling)
        
    elif kernel.WhichOneof('kernel_variant') == 'gemm_input1_input2':
        outer.append(kernel.gemm_input1_input2.outer)
        M.append(kernel.gemm_input1_input2.M)
        K.append(kernel.gemm_input1_input2.K)
        N.append(kernel.gemm_input1_input2.N)
        
        input_tensor_1_size.append(kernel.gemm_input1_input2.input_tensor_1_size)
        input_tensor_2_size.append(kernel.gemm_input1_input2.input_tensor_2_size)
        weight_tensor_size.append(-1.0)
        output_tensor_size.append(kernel.gemm_input1_input2.output_tensor_size)
        
        tiling.append(kernel.gemm_input1_input2.tiling)
            
    elif kernel.WhichOneof('kernel_variant') == 'elementwise_input1':
        outer.append(kernel.elementwise_input1.outer)
        M.append(kernel.elementwise_input1.M)
        K.append(1)
        N.append(kernel.elementwise_input1.N)
        
        input_tensor_1_size.append(kernel.elementwise_input1.input_tensor_size)
        input_tensor_2_size.append(-1.0)
        weight_tensor_size.append(-1.0)
        output_tensor_size.append(kernel.elementwise_input1.output_tensor_size)
        
        tiling.append(kernel.elementwise_input1.tiling)
        
    elif kernel.WhichOneof('kernel_variant') == 'elementwise_input1_input2':
        outer.append(kernel.elementwise_input1_input2.outer)
        M.append(kernel.elementwise_input1_input2.M)
        K.append(1)
        N.append(kernel.elementwise_input1_input2.N)
        
        input_tensor_1_size.append(kernel.elementwise_input1_input2.input_tensor_1_size)
        input_tensor_2_size.append(kernel.elementwise_input1_input2.input_tensor_2_size)
        weight_tensor_size.append(-1.0)
        output_tensor_size.append(kernel.elementwise_input1_input2.output_tensor_size)
        
        tiling.append(kernel.elementwise_input1_input2.tiling)

    else:
        raise Exception('Wrong!')
    
    node_dict[kernel.id] = i
    i += 1


startIdx = [] # upstream node id
endIdx = [] # upstream node id
edge_dict = {}
i = 0
for connection in dse.dataflow_graph.connections:
    startIdx.append(connection.startIdx)
    endIdx.append(connection.endIdx)
    edge_dict[(connection.startIdx, connection.endIdx)] = i
    i += 1


# assign edge tensor size
tensor_size = []
for connection in dse.dataflow_graph.connections:
    connection.tensor_size = output_tensor_size[node_dict[connection.startIdx]]
    tensor_size.append(connection.tensor_size)

num_kernel = len(kernel_name)
num_edge = len(startIdx)



# set buffer depth
kernel_topological_dict = {}
kernel_id_list = copy.deepcopy(list(node_dict.keys()))

edge_start_dict = {}
for connection in dse.dataflow_graph.connections:
    if connection.startIdx not in edge_start_dict.keys():
        edge_start_dict[connection.startIdx] = [connection.endIdx]
    else:
        edge_start_dict[connection.startIdx].append(connection.endIdx)

indegree = {}
for id in kernel_id_list:
    indegree[id] = 0
for start, end in edge_dict.keys():
    indegree[end] += 1


cnt = 0
while len(indegree.keys()) > 0:
    tmp = []
    for id in kernel_id_list:
        if id in indegree.keys() and indegree[id] == 0:
            kernel_topological_dict[id] = cnt
            del indegree[id]
            tmp.append(id)

    for id in tmp:
        if id in edge_start_dict.keys():
            for next_node_id in edge_start_dict[id]:
                indegree[next_node_id] -= 1
    cnt += 1




for kernel in dse.dataflow_graph.kernels:
    kernel.topological_number = kernel_topological_dict[kernel.id]

for connection in dse.dataflow_graph.connections:
    connection.buffer_depth = kernel_topological_dict[connection.endIdx] - kernel_topological_dict[connection.startIdx] + 1

startName = []
endName = []
i = 0
for connection in dse.dataflow_graph.connections:
    connection.startName = kernel_name[node_dict[startIdx[i]]]
    connection.endName = kernel_name[node_dict[endIdx[i]]]
    startName.append(connection.startName)
    endName.append(connection.endName)
    i += 1




last_fwd_kernel = -1
first_bwd_kernel = -1
for i in range(num_kernel):
    if fwd_bwd[i] == FWD_BWD.FWD.value:
        last_fwd_kernel = i

for i in range(num_kernel):       
    if fwd_bwd[i] == FWD_BWD.BWD.value:
        first_bwd_kernel = i
        break



Core = dse.system.accelerator.core
VecWidth = dse.system.accelerator.systolic_width
StageWidth = dse.system.accelerator.systolic_height
Freq = dse.system.accelerator.freq
num_chip = dse.system.num_chip
GFLOPS = 2*VecWidth*StageWidth*Core*Freq







if dse.execution.WhichOneof('workload_variant') != 'llm' or dse.execution.skip_inter_chip_optimization:
    sharding = []
    for i in range(5*num_kernel):
        sharding.append(Dim.NO_DIM.value)
        
    communication_size = []
    communication_type = []
    for i in range(num_kernel):
        communication_size.append(0)
        communication_type.append(Communication.NO_COMMUNICATION.value)

    edge_communication_size = []
    edge_communication_type = []
    for i in range(num_edge):
        edge_communication_size.append(0)
        edge_communication_type.append(Communication.NO_COMMUNICATION.value)
    
else:
    if dse.execution.llm.num_layer_in_graph == 0:
        raise Exception('Wrong!')
        
    if dse.execution.llm.num_layer_in_graph == 1: # assume users inputs only the dataflow graph of one layer, but there are many other layers denoted in "num_layer", which will be divided into PP (pipeline parallelism)
    
        model = Highs()
        # HiGHS does not support the same parameter settings; adjust if needed

        sharding = model.add_binary_matrix(num_kernel, 5, name='sharding')  
        communication_type = model.add_integer_vector(num_kernel, name='communication_type', lb=Communication.NO_COMMUNICATION.value, ub=Communication.ALL_GATHER.value)
        communication_size = model.add_continuous_vector(num_kernel, name='communication_size', lb=0)

        for i in range(num_kernel):
            if outer[i] == 1:
                model.add_constraint(sharding[i, Dim.OUTER.value] == 0)
            if M[i] == 1:
                model.add_constraint(sharding[i, Dim.M.value] == 0)
            if K[i] == 1:
                model.add_constraint(sharding[i, Dim.K.value] == 0)
            if N[i] == 1:
                model.add_constraint(sharding[i, Dim.N.value] == 0)
            model.add_constraint(binary_sum(sharding[i, :]) == 1)
                
            # don't shard the tile dim
            if tiling[i] == Dim.OUTER.value+1:
                model.add_constraint(sharding[i, Dim.OUTER.value] == 0)
            elif tiling[i] == Dim.M.value+1:
                model.add_constraint(sharding[i, Dim.M.value] == 0)
            elif tiling[i] == Dim.K.value+1:
                model.add_constraint(sharding[i, Dim.K.value] == 0)
            elif tiling[i] == Dim.N.value+1:
                model.add_constraint(sharding[i, Dim.N.value] == 0)
            else:
                raise Exception('Wrong!')
            
            

            if weight_tensor_size[i] == -1: # no weights
                pass
            else:
                model.add_constraint(sharding[i, Dim.NO_DIM.value] == 0)



            # if K is sharded
            model.add_constraint((sharding[i, Dim.K.value] == 1) >> (communication_type[i] == Communication.ALL_REDUCE.value))
            model.add_constraint((sharding[i, Dim.K.value] == 1) >> (communication_size[i] == output_tensor_size[i]))

            # if K is not sharded
            model.add_constraint((sharding[i, Dim.K.value] == 0) >> (communication_type[i] == Communication.NO_COMMUNICATION.value))
            model.add_constraint((sharding[i, Dim.K.value] == 0) >> (communication_size[i] == 0))

            if outer[i] > 1:
                model.add_constraint(sharding[i, Dim.OUTER.value] == 1)




            


        upstream_sharding = model.add_binary_matrix(num_edge, 3, name='upstream_sharding')
        downstream_sharding = model.add_binary_matrix(num_edge, 3, name='downstream_sharding')
        for i in range(num_edge):
            model.add_constraint(binary_sum(upstream_sharding[i, :]) == 1)
            model.add_constraint(binary_sum(downstream_sharding[i, :]) == 1)
            
            upstream_node_idx = node_dict[startIdx[i]]
            downstream_node_idx = node_dict[endIdx[i]]
            
            
            
            
            # upsteam 
            model.add_constraint((sharding[upstream_node_idx, Dim.OUTER.value] == 1) >> (upstream_sharding[i, Tensor.SR.value] == 1)) # shard outer
            model.add_constraint((sharding[upstream_node_idx, Dim.M.value] == 1) >> (upstream_sharding[i, Tensor.SR.value] == 1)) # shard M
            model.add_constraint((sharding[upstream_node_idx, Dim.K.value] == 1) >> (upstream_sharding[i, Tensor.RR.value] == 1)) # shard K
            model.add_constraint((sharding[upstream_node_idx, Dim.N.value] == 1) >> (upstream_sharding[i, Tensor.RS.value] == 1)) # shard N
            model.add_constraint((sharding[upstream_node_idx, Dim.NO_DIM.value] == 1) >> (upstream_sharding[i, Tensor.RR.value] == 1)) # no sharding



            # downstream
            if tiling[downstream_node_idx] == Dim.K.value+1: # for weight update kernels
                model.add_constraint((sharding[downstream_node_idx, Dim.OUTER.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1))
                model.add_constraint((sharding[downstream_node_idx, Dim.M.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1))
                model.add_constraint((sharding[downstream_node_idx, Dim.K.value] == 1) >> (downstream_sharding[i, Tensor.RS.value] == 1))
                model.add_constraint((sharding[downstream_node_idx, Dim.N.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1))
                model.add_constraint((sharding[downstream_node_idx, Dim.NO_DIM.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1))
            
            else:
                if kernel_type[downstream_node_idx] == KernelType.SIMD.value:
                    model.add_constraint((sharding[downstream_node_idx, Dim.OUTER.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard outer
                    model.add_constraint((sharding[downstream_node_idx, Dim.M.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard M
                    model.add_constraint((sharding[downstream_node_idx, Dim.K.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # shard K
                    model.add_constraint((sharding[downstream_node_idx, Dim.N.value] == 1) >> (downstream_sharding[i, Tensor.RS.value] == 1)) # shard N
                    model.add_constraint((sharding[downstream_node_idx, Dim.NO_DIM.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # no sharding

                else:
                    if weight_tensor_size[downstream_node_idx] != -1: # weight is present, this edge represents outer,K,N
                        model.add_constraint((sharding[downstream_node_idx, Dim.OUTER.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard outer
                        model.add_constraint((sharding[downstream_node_idx, Dim.M.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # shard M
                        model.add_constraint((sharding[downstream_node_idx, Dim.K.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard K
                        model.add_constraint((sharding[downstream_node_idx, Dim.N.value] == 1) >> (downstream_sharding[i, Tensor.RS.value] == 1)) # shard N
                        model.add_constraint((sharding[downstream_node_idx, Dim.NO_DIM.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # no sharding

                    else: # weight is not present
                        if tensor_size[i] == input_tensor_1_size[downstream_node_idx]: # if the edge represent tensor 1
                            model.add_constraint((sharding[downstream_node_idx, Dim.OUTER.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard outer
                            model.add_constraint((sharding[downstream_node_idx, Dim.M.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # shard M
                            model.add_constraint((sharding[downstream_node_idx, Dim.K.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard K
                            model.add_constraint((sharding[downstream_node_idx, Dim.N.value] == 1) >> (downstream_sharding[i, Tensor.RS.value] == 1)) # shard N
                            model.add_constraint((sharding[downstream_node_idx, Dim.NO_DIM.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # no sharding

                        elif tensor_size[i] == input_tensor_2_size[downstream_node_idx]: # if the edge represent tensor 2
                            model.add_constraint((sharding[downstream_node_idx, Dim.OUTER.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard outer
                            model.add_constraint((sharding[downstream_node_idx, Dim.M.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard M
                            model.add_constraint((sharding[downstream_node_idx, Dim.K.value] == 1) >> (downstream_sharding[i, Tensor.RS.value] == 1)) # shard K
                            model.add_constraint((sharding[downstream_node_idx, Dim.N.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # shard N
                            model.add_constraint((sharding[downstream_node_idx, Dim.NO_DIM.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # no sharding

                        else:
                            raise Exception('Wrong!')


        matrix_commu_type = [[0, 0, 0],
                  [Communication.ALL_GATHER.value, 0, Communication.ALL_TO_ALL.value], 
                  [Communication.ALL_GATHER.value, Communication.ALL_TO_ALL.value, 0]];

        matrix_commu_size = [[0, 0, 0],
                  [1, 0, 1], 
                  [1, 1, 0]];


        matrix_commu_type = np.array(matrix_commu_type)
        matrix_commu_size = np.array(matrix_commu_size)

        edge_communication_type = model.add_continuous_vector(num_edge, name='edge_communication_type', lb=0)
        edge_communication_size = model.add_continuous_vector(num_edge, name='edge_communication_size', lb=0)
        for i in range(num_edge):
            model.add_constraint(edge_communication_type[i] == upstream_sharding[i, :] @ matrix_commu_type @ downstream_sharding[i, :])
            model.add_constraint(edge_communication_size[i] == upstream_sharding[i, :] @ matrix_commu_size @ downstream_sharding[i, :] * tensor_size[i])



        total_communication_size = model.add_continuous_var(name='total_communication_size', lb=0)
        model.add_constraint(total_communication_size == np.ones((num_kernel)) @ communication_size + np.ones((num_edge)) @ edge_communication_size)



        model.set_objective_minimize(total_communication_size)
        model.solve()
    

    else: # assume user inputs all layers in the dataflow graph, and we need to break down them into PP
        if dse.system.WhichOneof('topology_variant') == 'sw': # 1D SW
            topology = [BasicTopology.SW.value]
            link_bw = [dse.system.sw.link_bw_x]
            dimension = [dse.system.sw.x]
            par = [dse.system.sw.par_x]
                
        elif dse.system.WhichOneof('topology_variant') == 'fc': # 1D FC
            topology = [BasicTopology.FC.value]
            link_bw = [dse.system.fc.link_bw_x]
            dimension = [dse.system.fc.x]
            par = [dse.system.fc.par_x]
            
        elif dse.system.WhichOneof('topology_variant') == 'r': # 1D Ring
            topology = [BasicTopology.R.value]
            link_bw = [dse.system.r.link_bw_x]
            dimension = [dse.system.r.x]    
            par = [dse.system.r.par_x]
            
        elif dse.system.WhichOneof('topology_variant') == 'r_r': # 2D Torus
            topology = [BasicTopology.R.value, BasicTopology.R.value]
            link_bw = [dse.system.r_r.link_bw_x, dse.system.r_r.link_bw_y]
            dimension = [dse.system.r_r.x, dse.system.r_r.y]
            par = [dse.system.r_r.par_x, dse.system.r_r.par_y]

        elif dse.system.WhichOneof('topology_variant') == 'fc_fc': # 2D Dragonfly
            topology = [BasicTopology.FC.value, BasicTopology.FC.value]
            link_bw = [dse.system.fc_fc.link_bw_x, dse.system.fc_fc.link_bw_y]
            dimension = [dse.system.fc_fc.x, dse.system.fc_fc.y]
            par = [dse.system.fc_fc.par_x, dse.system.fc_fc.par_y]
            
        elif dse.system.WhichOneof('topology_variant') == 'r_fc': # 2D ZionEX
            topology = [BasicTopology.R.value, BasicTopology.FC.value]
            link_bw = [dse.system.r_fc.link_bw_x, dse.system.r_fc.link_bw_y]
            dimension = [dse.system.r_fc.x, dse.system.r_fc.y]
            par = [dse.system.r_fc.par_x, dse.system.r_fc.par_y]

        elif dse.system.WhichOneof('topology_variant') == 'r_sw': # 2D DGX-1
            topology = [BasicTopology.R.value, BasicTopology.SW.value]
            link_bw = [dse.system.r_sw.link_bw_x, dse.system.r_sw.link_bw_y]
            dimension = [dse.system.r_sw.x, dse.system.r_sw.y]
            par = [dse.system.r_sw.par_x, dse.system.r_sw.par_y]
            
        elif dse.system.WhichOneof('topology_variant') == 'sw_sw': # 2D DGX-2
            topology = [BasicTopology.SW.value, BasicTopology.SW.value]
            link_bw = [dse.system.sw_sw.link_bw_x, dse.system.sw_sw.link_bw_y]
            dimension = [dse.system.sw_sw.x, dse.system.sw_sw.y]
            par = [dse.system.sw_sw.par_x, dse.system.sw_sw.par_y]
                
        elif dse.system.WhichOneof('topology_variant') == 'r_r_r': # 3D Torus
            topology = [BasicTopology.R.value, BasicTopology.R.value, BasicTopology.R.value]
            link_bw = [dse.system.r_r_r.link_bw_x, dse.system.r_r_r.link_bw_y, dse.system.r_r_r.link_bw_z]
            dimension = [dse.system.r_r_r.x, dse.system.r_r_r.y, dse.system.r_r_r.z]
            par = [dse.system.r_r_r.par_x, dse.system.r_r_r.par_y, dse.system.r_r_r.par_z]
            
        elif dse.system.WhichOneof('topology_variant') == 'r_sw_sw': # 3D DGX-1
            topology = [BasicTopology.R.value, BasicTopology.SW.value, BasicTopology.SW.value]
            link_bw = [dse.system.r_sw_sw.link_bw_x, dse.system.r_sw_sw.link_bw_y, dse.system.r_sw_sw.link_bw_z]
            dimension = [dse.system.r_sw_sw.x, dse.system.r_sw_sw.y, dse.system.r_sw_sw.z]
            par = [dse.system.r_sw_sw.par_x, dse.system.r_sw_sw.par_y, dse.system.r_sw_sw.par_z]
            
        elif dse.system.WhichOneof('topology_variant') == 'sw_sw_sw': # 3D DGX-2
            topology = [BasicTopology.SW.value, BasicTopology.SW.value, BasicTopology.SW.value]
            link_bw = [dse.system.sw_sw_sw.link_bw_x, dse.system.sw_sw_sw.link_bw_y, dse.system.sw_sw_sw.link_bw_z]
            dimension = [dse.system.sw_sw_sw.x, dse.system.sw_sw_sw.y, dse.system.sw_sw_sw.z]
            par = [dse.system.sw_sw_sw.par_x, dse.system.sw_sw_sw.par_y, dse.system.sw_sw_sw.par_z]

        else:
            raise Exception('Wrong!')


        





        # get PP number
        if len(topology) == 1: # 1D
            if par[0] == 'TP':
                num_partition = 1
            elif par[0] == 'PP':
                num_partition = dimension[0]
            elif par[0] == 'DP':
                num_partition = 1
            else:
                raise Exception('Wrong!')
                
        elif len(topology) == 2: # 2D
            if par[0] == 'TP' and par[1] == 'PP':
                num_partition = dimension[1]
            elif par[0] == 'TP' and par[1] == 'DP':
                num_partition = 1
            elif par[0] == 'PP' and par[1] == 'DP':
                num_partition = dimension[0]
            else:    
                raise Exception('Wrong!')
                
        elif len(topology) == 3: # 3D
            if par[0] == 'TP' and par[1] == 'PP' and par[2] == 'DP':
                num_partition = dimension[1]
            else:
                raise Exception('Wrong!')

        model = Highs()
        # HiGHS does not support the same parameter settings; adjust if needed
        
        
        
        
        
        sharding = model.add_binary_matrix(num_kernel, 5, name='sharding')  
        communication_type = model.add_integer_vector(num_kernel, name='communication_type', lb=Communication.NO_COMMUNICATION.value, ub=Communication.ALL_GATHER.value)
        communication_size = model.add_continuous_vector(num_kernel, name='communication_size', lb=0)
        
        C = num_partition
            

        A = model.add_binary_matrix(num_kernel, C, name='A') 
        H = model.add_binary_matrix(num_edge, C, name='H') 
        L = model.add_binary_matrix(num_edge, C, name='L') 
        Config = model.add_integer_vector(num_kernel, name='Config', lb=0)
        
        
        
        if dse.execution.execution_style == Execution_Style.KERNEL_BY_KERNEL.value:
            for i in range(len(configs)):
                model.add_constraint(Config[i] == i) # tuning nobe
            
        elif dse.execution.execution_style == Execution_Style.DATAFLOW.value:
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
                
                for i in range(num_kernel):
                    if fwd_bwd[i] == FWD_BWD.FWD.value:
                        model.add_constraint(Config[i] <= fwd_idx)
                    elif fwd_bwd[i] == FWD_BWD.BWD.value:
                        model.add_constraint(Config[i] >= bwd_idx)
                    else:
                        raise Exception('Wrong!')
                
        else:
            raise Exception('Wrong!')





        # kernel assignment   
        for i in range(num_kernel):
            model.add_constraint(A[i, :] @ np.ones((C)) == 1)
            
            
        t2 = np.zeros((C))
        for i in range(C):
            t2[i] = i
        for i in range(num_kernel):
            model.add_constraint(A[i, :] @ t2 == Config[i])


        for i in range(num_edge):
            model.add_constraint(Config[node_dict[startIdx[i]]] <= Config[node_dict[endIdx[i]]])
            
            
        
        
        
        if dse.execution.execution_style == Execution_Style.KERNEL_BY_KERNEL.value:
            for i in range(C):
                model.add_constraint(np.ones((num_kernel)) @ A[:, i] >= 1)
            model.add_constraint(C == num_kernel)
                
        elif dse.execution.execution_style == Execution_Style.DATAFLOW.value:
            pass
            
        else:
            raise Exception('Wrong!')
            
          
        
        
        # fill in H matrix
        for i in range(num_edge):
            start_node_idx = node_dict[startIdx[i]]
            end_node_idx = node_dict[endIdx[i]]
            
            for j in range(C):
                model.add_constraint(H[i, j] == A[start_node_idx, j])
        
        
        
        # ls, lt, utilities for E matrix
        ls = np.zeros(shape=(C, C))
        lt = np.zeros(shape=(C, C))

        for i in range(C):
            for j in range(C):
                if i <= j:
                    ls[i, j] = 1
                if i < j:
                    lt[i, j] = 1
            
            
        # fill in L matrix
        for i in range(num_edge):
            start_node_idx = node_dict[startIdx[i]]
            end_node_idx = node_dict[endIdx[i]]
            
            tmp1 = model.add_binary_vector(C)
            tmp2 = model.add_binary_vector(C)

            model.add_constraint(tmp1 == A[start_node_idx, :] @ ls)
            model.add_constraint(tmp2 == A[end_node_idx, :] @ lt)
            
            for j in range(C):
                t1 = model.add_binary_var()
                t2 = model.add_binary_var()
                t3 = model.add_binary_var()
                t4 = model.add_binary_var()
                t5 = model.add_binary_var()
                t6 = model.add_binary_var()
                t7 = model.add_binary_var()
                t8 = model.add_binary_var()
                t9 = model.add_binary_var()
                t10 = model.add_binary_var()
                t11 = model.add_binary_var()

                model.add_constraint(t1 == binary_and(tmp1[j], tmp2[j]))
                model.add_constraint(t2 == binary_or(tmp1[j], tmp2[j]))
                model.add_constraint(t3 == 1 - t1)
                model.add_constraint(t4 == binary_and(t2, t3))
                
                model.add_constraint(t5 == binary_and(A[start_node_idx, j], A[end_node_idx, j]))
                
                model.add_constraint(t6 == binary_and(t4, t5))
                model.add_constraint(t7 == binary_or(t4, t5))
                model.add_constraint(t8 == 1 - t6)
                model.add_constraint(t9 == binary_and(t7, t8))
                model.add_constraint(t10 == 1 - A[end_node_idx, j])
                model.add_constraint(t11 == binary_and(t9, t10))
                
                model.add_constraint((t11 == 1) >> (L[i, j] == 1))
                model.add_constraint((t11 == 0) >> (L[i, j] == 0))
        
        
        

        
        
        # compute resources
        if dse.execution.compute_util == 0:
            Par_lane = model.add_integer_vector(num_kernel, name='Par_lane', lb=1)
            Par_stage = model.add_integer_vector(num_kernel, name='Par_stage', lb=1)
        else:
            pass

        Par_total = model.add_integer_vector(num_kernel, name='Par_total', lb=1)

        if dse.execution.compute_util == 0:
            for i in range(num_kernel):
                model.add_constraint(Par_lane[i] * Par_stage[i] == Par_total[i])
        else:
            pass
            
        for i in range(C):
            model.add_constraint(Par_total @ A[:, i] == Core)        



        
        
        





        ALL_REDUCE_ratio = model.add_continuous_var(name='ALL_REDUCE_ratio', lb=0)
        ALL_TO_ALL_ratio = model.add_continuous_var(name='ALL_TO_ALL_ratio', lb=0)
        ALL_GATHER_ratio = model.add_continuous_var(name='ALL_GATHER_ratio', lb=0)
        TP = model.add_integer_var(name='TP', lb=1)
        PP = model.add_integer_var(name='PP', lb=1)
        DP = model.add_integer_var(name='DP', lb=1)

        if len(topology) == 1: # 1D
            Shape = model.add_integer_vector(1, name='Shape', lb=0)
            model.add_constraint(Shape[0] == num_chip)
            
            Link_BW = model.add_continuous_vector(1, name='Link_BW', lb=0)
            for i in range(len(topology)):
                model.add_constraint(Link_BW[i] == link_bw[i])  
                
            Link_BW_TP = model.add_continuous_var(name='Link_BW_TP', lb=0)
            Link_BW_PP = model.add_continuous_var(name='Link_BW_PP', lb=0)
            Link_BW_DP = model.add_continuous_var(name='Link_BW_DP', lb=0)    
            
            if par[0] == 'TP':
                model.add_constraint(TP == num_chip)
                model.add_constraint(PP == 1)
                model.add_constraint(DP == 1)   

                model.add_constraint(Link_BW_TP == Link_BW[0])
                
                aaa = model.add_continuous_var()
                model.add_constraint(aaa == TP * Link_BW_TP)
                if topology[0] == BasicTopology.R.value:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == TP - 1)
                    model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP * 8 == TP * TP)
                    model.add_constraint(ALL_GATHER_ratio * aaa == TP - 1)
                elif topology[0] == BasicTopology.FC.value:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == 1)
                    model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP == 1)
                    model.add_constraint(ALL_GATHER_ratio * aaa == 1)
                elif topology[0] == BasicTopology.SW.value:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == TP - 1)
                    model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP * 4 == TP)
                    model.add_constraint(ALL_GATHER_ratio * aaa == TP - 1)
                else:
                    raise Exception('Wrong!')
                
            elif par[0] == 'PP':
                model.add_constraint(TP == 1)
                model.add_constraint(PP == num_chip)
                model.add_constraint(DP == 1)   

                model.add_constraint(Link_BW_PP == Link_BW[0])
                
                model.add_constraint(ALL_REDUCE_ratio == 0)
                model.add_constraint(ALL_TO_ALL_ratio == 0)
                model.add_constraint(ALL_GATHER_ratio == 0)
                
            elif par[0] == 'DP':
                model.add_constraint(TP == 1)
                model.add_constraint(PP == 1)
                model.add_constraint(DP == num_chip)   

                model.add_constraint(Link_BW_DP == Link_BW[0])
                
                model.add_constraint(ALL_REDUCE_ratio == 0)
                model.add_constraint(ALL_TO_ALL_ratio == 0)
                model.add_constraint(ALL_GATHER_ratio == 0)
                
            else:
                raise Exception('Wrong!')
                
        elif len(topology) == 2: # 2D
            Shape = model.add_integer_vector(2, name='Shape', lb=1)
            if dimension[0] == 0:
                pass
            else:
                model.add_constraint(Shape[0] == dimension[0])
            
            if dimension[1] == 0:
                pass
            else:
                model.add_constraint(Shape[1] == dimension[1])
                
            model.add_constraint(Shape[0] * Shape[1] == num_chip)
            
            Link_BW = model.add_continuous_vector(2, name='Link_BW', lb=0)
            for i in range(len(topology)):
                model.add_constraint(Link_BW[i] == link_bw[i])
                
            Link_BW_TP = model.add_continuous_var(name='Link_BW_TP', lb=0)
            Link_BW_PP = model.add_continuous_var(name='Link_BW_PP', lb=0)
            Link_BW_DP = model.add_continuous_var(name='Link_BW_DP', lb=0)
            
            if par[0] == 'TP' and par[1] == 'PP':
                model.add_constraint(TP == Shape[0])
                model.add_constraint(PP == Shape[1])
                model.add_constraint(DP == 1)

                model.add_constraint(Link_BW_TP == Link_BW[0])
                model.add_constraint(Link_BW_PP == Link_BW[1])
                
                aaa = model.add_continuous_var()
                model.add_constraint(aaa == TP * Link_BW_TP)
                if topology[0] == BasicTopology.R.value:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == TP - 1)
                    model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP * 8 == TP * TP)
                    model.add_constraint(ALL_GATHER_ratio * aaa == TP - 1)
                elif topology[0] == BasicTopology.FC.value:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == 1)
                    model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP == 1)
                    model.add_constraint(ALL_GATHER_ratio * aaa == 1)
                elif topology[0] == BasicTopology.SW.value:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == TP - 1)
                    model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP * 4 == TP)
                    model.add_constraint(ALL_GATHER_ratio * aaa == TP - 1)
                else:
                    raise Exception('Wrong!')
                
            elif par[0] == 'TP' and par[1] == 'DP':
                model.add_constraint(TP == Shape[0])
                model.add_constraint(PP == 1)
                model.add_constraint(DP == Shape[1])

                model.add_constraint(Link_BW_TP == Link_BW[0])
                model.add_constraint(Link_BW_DP == Link_BW[1])

                aaa = model.add_continuous_var()
                model.add_constraint(aaa == TP * Link_BW_TP)
                if topology[0] == BasicTopology.R.value:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == TP - 1)
                    model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP * 8 == TP * TP)
                    model.add_constraint(ALL_GATHER_ratio * aaa == TP - 1)
                elif topology[0] == BasicTopology.FC.value:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == 1)
                    model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP == 1)
                    model.add_constraint(ALL_GATHER_ratio * aaa == 1)
                elif topology[0] == BasicTopology.SW.value:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == TP - 1)
                    model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP * 4 == TP)
                    model.add_constraint(ALL_GATHER_ratio * aaa == TP - 1)
                else:
                    raise Exception('Wrong!')
                    
            elif par[0] == 'PP' and par[1] == 'DP':
                model.add_constraint(TP == 1)
                model.add_constraint(PP == Shape[0])
                model.add_constraint(DP == Shape[1])

                model.add_constraint(Link_BW_PP == Link_BW[0])
                model.add_constraint(Link_BW_DP == Link_BW[1])
                
                model.add_constraint(ALL_REDUCE_ratio == 0)
                model.add_constraint(ALL_TO_ALL_ratio == 0)
                model.add_constraint(ALL_GATHER_ratio == 0)
                
            else:    
                raise Exception('Wrong!')
                
        elif len(topology) == 3: # 3D
            Shape = model.add_integer_vector(3, name='Shape', lb=1)
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
                
            aaa = model.add_continuous_var()
            model.add_constraint(aaa == Shape[0] * Shape[1])
            model.add_constraint(aaa * Shape[2] == num_chip)
            
            
            Link_BW = model.add_continuous_vector(3, name='Link_BW', lb=0)
            for i in range(len(topology)):
                model.add_constraint(Link_BW[i] == link_bw[i])  

            Link_BW_TP = model.add_continuous_var(name='Link_BW_TP', lb=0)
            Link_BW_PP = model.add_continuous_var(name='Link_BW_PP', lb=0)
            Link_BW_DP = model.add_continuous_var(name='Link_BW_DP', lb=0)
            
            if par[0] == 'TP' and par[1] == 'PP' and par[2] == 'DP':
                model.add_constraint(TP == Shape[0])
                model.add_constraint(PP == Shape[1])
                model.add_constraint(DP == Shape[2])
                
                model.add_constraint(Link_BW_TP == Link_BW[0])
                model.add_constraint(Link_BW_PP == Link_BW[1])
                model.add_constraint(Link_BW_DP == Link_BW[2])

                aaa = model.add_continuous_var()
                model.add_constraint(aaa == TP * Link_BW_TP)
                if topology[0] == BasicTopology.R.value:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == TP - 1)
                    model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP * 8 == TP * TP)
                    model.add_constraint(ALL_GATHER_ratio * aaa == TP - 1)
                elif topology[0] == BasicTopology.FC.value:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == 1)
                    model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP == 1)
                    model.add_constraint(ALL_GATHER_ratio * aaa == 1)
                elif topology[0] == BasicTopology.SW.value:
                    model.add_constraint(ALL_REDUCE_ratio * aaa == TP - 1)
                    model.add_constraint(ALL_TO_ALL_ratio * Link_BW_TP * 4 == TP)
                    model.add_constraint(ALL_GATHER_ratio * aaa == TP - 1)
                else:
                    raise Exception('Wrong!')
                


                
            else:
                raise Exception('Wrong!')
            
        else:
            raise Exception('Wrong!')












        M_sharded = model.add_continuous_vector(num_kernel, name='M_sharded', lb=1)
        for i in range(num_kernel):
            model.add_constraint(M_sharded[i] * TP >= M[i])
        
        # compute cycle
        if dse.execution.compute_util == 0:
            Cycle = model.add_integer_vector(num_kernel, name='Cycle', lb=0)
            m_factor = model.add_integer_vector(num_kernel, name='m_factor', lb=1)
            n_factor = model.add_integer_vector(num_kernel, name='n_factor', lb=1)
        else:
            FLOP_per_kernel = model.add_integer_vector(num_kernel, name='FLOP_per_kernel', lb=0)

        for i in range(num_kernel):
            if kernel_type[i] == KernelType.SIMD.value:
                if dse.execution.compute_util == 0:
                    model.add_constraint(m_factor[i] * Par_lane[i] * VecWidth >= M_sharded[i])
                    model.add_constraint(Par_stage[i] == 1)
                    model.add_constraint(Cycle[i] == m_factor[i] * N[i])
                else:
                    model.add_constraint(FLOP_per_kernel[i] == M_sharded[i] * N[i])
                
            elif kernel_type[i] == KernelType.SYSTOLIC.value:
                if dse.execution.compute_util == 0:
                    model.add_constraint(m_factor[i] * Par_lane[i] * VecWidth >= M_sharded[i])
                    model.add_constraint(n_factor[i] * Par_stage[i] * StageWidth >= N[i])

                    tmp = model.add_integer_var(lb=0)
                    model.add_constraint(tmp == m_factor[i] * n_factor[i])
                    model.add_constraint(Cycle[i] == tmp * K[i])
                        
                else:
                    aaa = model.add_integer_var(lb=0)
                    model.add_constraint(aaa == 2 * M_sharded[i] * N[i])
                    model.add_constraint(FLOP_per_kernel[i] == aaa * K[i])
                    
            else:
                raise Exception('Wrong!')

        
        
        
        
        
        compute_latency_per_partition = model.add_continuous_vector(C, name='compute_latency_per_partition', lb=0)
        for i in range(C):
            if dse.execution.compute_util == 0:
                t1 = model.add_integer_vector(num_kernel, lb=0)
                t2 = model.add_continuous_var()
                for j in range(num_kernel):
                    model.add_constraint(t1[j] == Cycle[j] * A[j, i])
                model.add_constraint(t2 == max(t1[j] for j in range(num_kernel)))
                model.add_constraint(compute_latency_per_partition[i] == t2 / Freq)
                
            else:
                t3 = model.add_integer_vector(num_kernel, lb=0)
                t4 = model.add_continuous_var()
                for j in range(num_kernel):
                    model.add_constraint(t3[j] == FLOP_per_kernel[j] * A[j, i])
                model.add_constraint(t4 == t3 @ np.ones((num_kernel)))
                
                tmp = 1 / dse.execution.compute_util
                model.add_constraint(compute_latency_per_partition[i] == t4 * tmp / GFLOPS)
            
            
            
        
        
        


        for i in range(num_kernel):
            if outer[i] == 1:
                model.add_constraint(sharding[i, Dim.OUTER.value] == 0)
            if M[i] == 1:
                model.add_constraint(sharding[i, Dim.M.value] == 0)
            if K[i] == 1:
                model.add_constraint(sharding[i, Dim.K.value] == 0)
            if N[i] == 1:
                model.add_constraint(sharding[i, Dim.N.value] == 0)
            model.add_constraint(binary_sum(sharding[i, :]) == 1)
                
            # don't shard the tile dim
            if tiling[i] == Dim.OUTER.value+1:
                model.add_constraint(sharding[i, Dim.OUTER.value] == 0)
            elif tiling[i] == Dim.M.value+1:
                model.add_constraint(sharding[i, Dim.M.value] == 0)
            elif tiling[i] == Dim.K.value+1:
                model.add_constraint(sharding[i, Dim.K.value] == 0)
            elif tiling[i] == Dim.N.value+1:
                model.add_constraint(sharding[i, Dim.N.value] == 0)
            else:
                raise Exception('Wrong!')
            
            

            if weight_tensor_size[i] == -1: # no weights
                pass
            else:
                model.add_constraint(sharding[i, Dim.NO_DIM.value] == 0)



            # if K is sharded
            model.add_constraint((sharding[i, Dim.K.value] == 1) >> (communication_type[i] == Communication.ALL_REDUCE.value))
            model.add_constraint((sharding[i, Dim.K.value] == 1) >> (communication_size[i] == output_tensor_size[i]))

            # if K is not sharded
            model.add_constraint((sharding[i, Dim.K.value] == 0) >> (communication_type[i] == Communication.NO_COMMUNICATION.value))
            model.add_constraint((sharding[i, Dim.K.value] == 0) >> (communication_size[i] == 0))

            if outer[i] > 1:
                model.add_constraint(sharding[i, Dim.OUTER.value] == 1)




            


        upstream_sharding = model.add_binary_matrix(num_edge, 3, name='upstream_sharding')
        downstream_sharding = model.add_binary_matrix(num_edge, 3, name='downstream_sharding')
        for i in range(num_edge):
            model.add_constraint(binary_sum(upstream_sharding[i, :]) == 1)
            model.add_constraint(binary_sum(downstream_sharding[i, :]) == 1)
            
            upstream_node_idx = node_dict[startIdx[i]]
            downstream_node_idx = node_dict[endIdx[i]]
            
            
            
            
            # upsteam 
            model.add_constraint((sharding[upstream_node_idx, Dim.OUTER.value] == 1) >> (upstream_sharding[i, Tensor.SR.value] == 1)) # shard outer
            model.add_constraint((sharding[upstream_node_idx, Dim.M.value] == 1) >> (upstream_sharding[i, Tensor.SR.value] == 1)) # shard M
            model.add_constraint((sharding[upstream_node_idx, Dim.K.value] == 1) >> (upstream_sharding[i, Tensor.RR.value] == 1)) # shard K
            model.add_constraint((sharding[upstream_node_idx, Dim.N.value] == 1) >> (upstream_sharding[i, Tensor.RS.value] == 1)) # shard N
            model.add_constraint((sharding[upstream_node_idx, Dim.NO_DIM.value] == 1) >> (upstream_sharding[i, Tensor.RR.value] == 1)) # no sharding



            # downstream
            if tiling[downstream_node_idx] == Dim.K.value+1: # for weight update kernels
                model.add_constraint((sharding[downstream_node_idx, Dim.OUTER.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1))
                model.add_constraint((sharding[downstream_node_idx, Dim.M.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1))
                model.add_constraint((sharding[downstream_node_idx, Dim.K.value] == 1) >> (downstream_sharding[i, Tensor.RS.value] == 1))
                model.add_constraint((sharding[downstream_node_idx, Dim.N.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1))
                model.add_constraint((sharding[downstream_node_idx, Dim.NO_DIM.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1))
            
            else:
                if kernel_type[downstream_node_idx] == KernelType.SIMD.value:
                    model.add_constraint((sharding[downstream_node_idx, Dim.OUTER.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard outer
                    model.add_constraint((sharding[downstream_node_idx, Dim.M.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard M
                    model.add_constraint((sharding[downstream_node_idx, Dim.K.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # shard K
                    model.add_constraint((sharding[downstream_node_idx, Dim.N.value] == 1) >> (downstream_sharding[i, Tensor.RS.value] == 1)) # shard N
                    model.add_constraint((sharding[downstream_node_idx, Dim.NO_DIM.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # no sharding

                else:
                    if weight_tensor_size[downstream_node_idx] != -1: # weight is present, this edge represents outer,K,N
                        model.add_constraint((sharding[downstream_node_idx, Dim.OUTER.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard outer
                        model.add_constraint((sharding[downstream_node_idx, Dim.M.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # shard M
                        model.add_constraint((sharding[downstream_node_idx, Dim.K.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard K
                        model.add_constraint((sharding[downstream_node_idx, Dim.N.value] == 1) >> (downstream_sharding[i, Tensor.RS.value] == 1)) # shard N
                        model.add_constraint((sharding[downstream_node_idx, Dim.NO_DIM.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # no sharding

                    else: # weight is not present
                        if tensor_size[i] == input_tensor_1_size[downstream_node_idx]: # if the edge represent tensor 1
                            model.add_constraint((sharding[downstream_node_idx, Dim.OUTER.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard outer
                            model.add_constraint((sharding[downstream_node_idx, Dim.M.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # shard M
                            model.add_constraint((sharding[downstream_node_idx, Dim.K.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard K
                            model.add_constraint((sharding[downstream_node_idx, Dim.N.value] == 1) >> (downstream_sharding[i, Tensor.RS.value] == 1)) # shard N
                            model.add_constraint((sharding[downstream_node_idx, Dim.NO_DIM.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # no sharding

                        elif tensor_size[i] == input_tensor_2_size[downstream_node_idx]: # if the edge represent tensor 2
                            model.add_constraint((sharding[downstream_node_idx, Dim.OUTER.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard outer
                            model.add_constraint((sharding[downstream_node_idx, Dim.M.value] == 1) >> (downstream_sharding[i, Tensor.SR.value] == 1)) # shard M
                            model.add_constraint((sharding[downstream_node_idx, Dim.K.value] == 1) >> (downstream_sharding[i, Tensor.RS.value] == 1)) # shard K
                            model.add_constraint((sharding[downstream_node_idx, Dim.N.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # shard N
                            model.add_constraint((sharding[downstream_node_idx, Dim.NO_DIM.value] == 1) >> (downstream_sharding[i, Tensor.RR.value] == 1)) # no sharding

                        else:
                            raise Exception('Wrong!')


        matrix_commu_type = [[0, 0, 0],
                  [Communication.ALL_GATHER.value, 0, Communication.ALL_TO_ALL.value], 
                  [Communication.ALL_GATHER.value, Communication.ALL_TO_ALL.value, 0]];

        matrix_commu_size = [[0, 0, 0],
                  [1, 0, 1], 
                  [1, 1, 0]];


        matrix_commu_type = np.array(matrix_commu_type)
        matrix_commu_size = np.array(matrix_commu_size)

        edge_communication_type = model.add_continuous_vector(num_edge, name='edge_communication_type', lb=0)
        edge_communication_size = model.add_continuous_vector(num_edge, name='edge_communication_size', lb=0)
        buffer_size = model.add_continuous_vector(num_edge, name='buffer_size', lb=0)
        for i in range(num_edge):
            model.add_constraint(edge_communication_type[i] == upstream_sharding[i, :] @ matrix_commu_type @ downstream_sharding[i, :])
            model.add_constraint(edge_communication_size[i] == upstream_sharding[i, :] @ matrix_commu_size @ downstream_sharding[i, :] * tensor_size[i])
            model.add_constraint(buffer_size[i] == tensor_size[i])













        Network_Latency_ALL_REDUCE_node = model.add_continuous_vector(C, name='Network_Latency_ALL_REDUCE_node', lb=0)
        for i in range(C):
            t1 = model.add_continuous_var()
            t2 = model.add_continuous_var()
            model.add_constraint(t1 == A[:, i] @ communication_size)
            model.add_constraint(t2 == ALL_REDUCE_ratio)
            model.add_constraint(Network_Latency_ALL_REDUCE_node[i] == t1*t2)


        Network_Latency_ALL_TO_ALL_node = model.add_continuous_vector(C, name='Network_Latency_ALL_TO_ALL_node', lb=0)
        for i in range(C):
            t1 = model.add_continuous_var()
            t2 = model.add_continuous_var()
            model.add_constraint(t1 == A[:, i] @ communication_size)
            model.add_constraint(t2 == ALL_TO_ALL_ratio)
            model.add_constraint(Network_Latency_ALL_TO_ALL_node[i] == t1*t2)


        Network_Latency_ALL_GATHER_node = model.add_continuous_vector(C, name='Network_Latency_ALL_GATHER_node', lb=0)
        for i in range(C):
            t1 = model.add_continuous_var()
            t2 = model.add_continuous_var()
            model.add_constraint(t1 == A[:, i] @ communication_size)
            model.add_constraint(t2 == ALL_GATHER_ratio)
            model.add_constraint(Network_Latency_ALL_GATHER_node[i] == t1*t2)






        Network_Latency_ALL_REDUCE_edge = model.add_continuous_vector(C, name='Network_Latency_ALL_REDUCE_edge', lb=0)
        for i in range(C):
            t1 = model.add_continuous_var()
            t2 = model.add_continuous_var()
            model.add_constraint(t1 == H[:, i] @ edge_communication_size)
            model.add_constraint(t2 == ALL_REDUCE_ratio)
            model.add_constraint(Network_Latency_ALL_REDUCE_edge[i] == t1*t2)


        Network_Latency_ALL_TO_ALL_edge = model.add_continuous_vector(C, name='Network_Latency_ALL_TO_ALL_edge', lb=0)
        for i in range(C):
            t1 = model.add_continuous_var()
            t2 = model.add_continuous_var()
            model.add_constraint(t1 == H[:, i] @ edge_communication_size)
            model.add_constraint(t2 == ALL_TO_ALL_ratio)
            model.add_constraint(Network_Latency_ALL_TO_ALL_edge[i] == t1*t2)

        Network_Latency_ALL_GATHER_edge = model.add_continuous_vector(C, name='Network_Latency_ALL_GATHER_edge', lb=0)
        for i in range(C):
            t1 = model.add_continuous_var()
            t2 = model.add_continuous_var()
            model.add_constraint(t1 == H[:, i] @ edge_communication_size)
            model.add_constraint(t2 == ALL_GATHER_ratio)
            model.add_constraint(Network_Latency_ALL_GATHER_edge[i] == t1*t2)



        p2p_communication_latency_per_partition = model.add_continuous_vector(C, name='p2p_communication_latency_per_partition', lb=0)
        
        for i in range(C):
            model.add_constraint(p2p_communication_latency_per_partition[i] * Link_BW_PP >= L[:, i] @ buffer_size)

            
        
        latency_per_partition = model.add_continuous_vector(C, name='latency_per_partition', lb=0)
        for i in range(C): 
            model.add_constraint(latency_per_partition[i] == Network_Latency_ALL_REDUCE_node[i]
                                                      + Network_Latency_ALL_TO_ALL_node[i]
                                                      + Network_Latency_ALL_GATHER_node[i]
                                                      + Network_Latency_ALL_REDUCE_edge[i]
                                                      + Network_Latency_ALL_TO_ALL_edge[i]
                                                      + Network_Latency_ALL_GATHER_edge[i]
                                                      + compute_latency_per_partition[i]
                                                      + p2p_communication_latency_per_partition[i])






        total_latency = model.add_continuous_var(name='total_latency', lb=0)
        model.add_constraint(total_latency == max(latency_per_partition[i] for i in range(C)))



        model.set_objective_minimize(total_latency)
        model.solve()




    # get variable values from HiGHS program
    sharding = []
    communication_size = []
    communication_type = []
    edge_communication_size = []
    edge_communication_type = []
    latency_per_partition = []
    A = []

    for v in model.get_solution():
        print(v.varName, v.value)

        if v.varName.startswith('sharding'):
            sharding.append(v.value)
        if v.varName.startswith('communication_size'):
            communication_size.append(v.value)
        if v.varName.startswith('communication_type'):
            communication_type.append(v.value)

        if v.varName.startswith('edge_communication_size'):
            edge_communication_size.append(v.value)
        if v.varName.startswith('edge_communication_type'):
            edge_communication_type.append(v.value)

        if v.varName.startswith('latency_per_partition'):
            latency_per_partition.append(v.value)

        if v.varName.startswith('A['):
            A.append(v.value)



       
        

    
    
    
# update kernels
i = 0
for kernel in dse.dataflow_graph.kernels:
    if kernel.WhichOneof('kernel_variant') == 'gemm_input1_weight':
        if sharding[i*5+0] == 1:
            kernel.gemm_input1_weight.sharding = Dim.OUTER.value+1
        elif sharding[i*5+1] == 1:
            kernel.gemm_input1_weight.sharding = Dim.M.value+1
        elif sharding[i*5+2] == 1:
            kernel.gemm_input1_weight.sharding = Dim.K.value+1
        elif sharding[i*5+3] == 1:
            kernel.gemm_input1_weight.sharding = Dim.N.value+1
        else:
            kernel.gemm_input1_weight.sharding = Dim.NO_DIM.value+1
         
        if kernel.gemm_input1_weight.communication_type != Communication.NO_COMMUNICATION.value:
            pass
        else:
            kernel.gemm_input1_weight.communication_size = float(communication_size[i])
            kernel.gemm_input1_weight.communication_type = int(communication_type[i])
        i += 1
        
    elif kernel.WhichOneof('kernel_variant') == 'gemm_input1_input2':
        if sharding[i*5+0] == 1:
            kernel.gemm_input1_input2.sharding = Dim.OUTER.value+1
        elif sharding[i*5+1] == 1:
            kernel.gemm_input1_input2.sharding = Dim.M.value+1
        elif sharding[i*5+2] == 1:
            kernel.gemm_input1_input2.sharding = Dim.K.value+1
        elif sharding[i*5+3] == 1:
            kernel.gemm_input1_input2.sharding = Dim.N.value+1
        else:
            kernel.gemm_input1_input2.sharding = Dim.NO_DIM.value+1
        
        if kernel.gemm_input1_input2.communication_type != Communication.NO_COMMUNICATION.value:
            pass
        else:
            kernel.gemm_input1_input2.communication_size = float(communication_size[i])
            kernel.gemm_input1_input2.communication_type = int(communication_type[i])
        i += 1
        
    elif kernel.WhichOneof('kernel_variant') == 'elementwise_input1':
        if sharding[i*5+0] == 1:
            kernel.elementwise_input1.sharding = Dim.OUTER.value+1
        elif sharding[i*5+1] == 1:
            kernel.elementwise_input1.sharding = Dim.M.value+1
        elif sharding[i*5+2] == 1:
            kernel.elementwise_input1.sharding = Dim.K.value+1
        elif sharding[i*5+3] == 1:
            kernel.elementwise_input1.sharding = Dim.N.value+1
        else:
            kernel.elementwise_input1.sharding = Dim.NO_DIM.value+1
        
        if kernel.elementwise_input1.communication_type != Communication.NO_COMMUNICATION.value:
            pass
        else:        
            kernel.elementwise_input1.communication_size = float(communication_size[i])
            kernel.elementwise_input1.communication_type = int(communication_type[i])
        i += 1
        
    elif kernel.WhichOneof('kernel_variant') == 'elementwise_input1_input2':
        if sharding[i*5+0] == 1:
            kernel.elementwise_input1_input2.sharding = Dim.OUTER.value+1
        elif sharding[i*5+1] == 1:
            kernel.elementwise_input1_input2.sharding = Dim.M.value+1
        elif sharding[i*5+2] == 1:
            kernel.elementwise_input1_input2.sharding = Dim.K.value+1
        elif sharding[i*5+3] == 1:
            kernel.elementwise_input1_input2.sharding = Dim.N.value+1
        else:
            kernel.elementwise_input1_input2.sharding = Dim.NO_DIM.value+1
            
        if kernel.elementwise_input1_input2.communication_type != Communication.NO_COMMUNICATION.value:
            pass
        else:
            kernel.elementwise_input1_input2.communication_size = float(communication_size[i])
            kernel.elementwise_input1_input2.communication_type = int(communication_type[i])
        i += 1
        
    else:
        raise Exception('Wrong!')


# update edges
i = 0
for connection in dse.dataflow_graph.connections:
    connection.communication_size = float(edge_communication_size[i])
    connection.communication_type = int(edge_communication_type[i])
    i += 1

if dse.execution.WhichOneof('workload_variant') == 'llm' and dse.execution.llm.num_layer_in_graph > 1:
    # get the longest partition
    maxV = max(latency_per_partition)
    idx = latency_per_partition.index(maxV)

    kernel_idx_to_keep = []
    kernel_idx_to_remove = []
    kernel_content_to_remove = []
    tensor_content_to_remove = []

    for i in range(num_kernel):
        for j in range(num_partition):
            if j == idx and A[i * num_partition + j] == 0:
                kernel_idx_to_remove.append(i)
            if j == idx and A[i * num_partition + j] == 1:
                kernel_idx_to_keep.append(i)
    i = 0
    for kernel in dse.dataflow_graph.kernels:
        if i in kernel_idx_to_remove:
            kernel_content_to_remove.append(copy.deepcopy(dse.dataflow_graph.kernels[i]))
        i += 1


    for tensor in dse.dataflow_graph.connections:
        start = tensor.startIdx
        end = tensor.endIdx
        if not node_dict[start] in kernel_idx_to_keep or not node_dict[end] in kernel_idx_to_keep:
            tensor_content_to_remove.append(copy.deepcopy(tensor))


    for content in kernel_content_to_remove:
        dse.dataflow_graph.kernels.remove(content)
        
    for content in tensor_content_to_remove:
        dse.dataflow_graph.connections.remove(content)

    
        

# write to sharded binary
with open('./'+name+'/'+'dse_sharded.pb', "wb") as file:
    file.write(dse.SerializeToString())


# write to sharded text file
with open('./'+name+'/'+'dse_sharded.txt', "w") as file:
    text_format.PrintMessage(dse, file)



# create dot graph
node_list = []
edge_list = []
dict = {}
graph = pydot.Dot(graph_type='digraph')
for kernel in dse.dataflow_graph.kernels:  
    label = text_format.MessageToString(kernel)
    pydot_node = pydot.Node(kernel.name, style="filled", fillcolor="white", label=label, penwidth=8)
    dict[kernel.id] = pydot_node
    graph.add_node(pydot_node)

for connection in dse.dataflow_graph.connections:
    label = text_format.MessageToString(connection)
    pydot_edge = pydot.Edge(dict[connection.startIdx], dict[connection.endIdx], label=label, penwidth=8)
    graph.add_edge(pydot_edge)


graph.write_png('./'+name+'/'+'dataflow_graph_sharded.png')
