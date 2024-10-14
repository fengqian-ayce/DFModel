import setup_pb2
import csv
import argparse
from google.protobuf import text_format
import pydot
import math


# user pass in
parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, required=True)
parser.add_argument('--mlp_dim', type=int, required=True)
parser.add_argument('--num_head', type=int, required=True)
parser.add_argument('--head_dim', type=int, required=True)
parser.add_argument('--seq', type=int, required=True)
parser.add_argument('--word', type=int, required=True)
args = parser.parse_args()


hidden = args.hidden
mlp_dim = args.mlp_dim
num_head = args.num_head
head_dim = args.head_dim
seq = args.seq
word = args.word



if num_head * head_dim != hidden:
    raise Exception('Wrong!')





# total 6*stage+3 kernels
stage = math.ceil(math.log2(seq))

def FFT(counter, name):
    for i in range(stage):
        kernel = dataflow_graph.kernels.add()
        kernel.name = name+'_stage'+str(i)
        kernel.id = counter+i
        kernel.fwd_bwd = 1
        kernel.type = 1
        kernel.config = -1
        kernel.gemm_input1_weight.outer = 1
        kernel.gemm_input1_weight.M = 2*seq
        kernel.gemm_input1_weight.K = 2
        kernel.gemm_input1_weight.N = 1
        kernel.gemm_input1_weight.input_tensor_size = 2*seq*word
        kernel.gemm_input1_weight.output_tensor_size = 2*seq*word
        kernel.gemm_input1_weight.tiling = 5
        kernel.gemm_input1_weight.skip_weight = True
        kernel.gemm_input1_weight.num_input = hidden
        kernel.gemm_input1_weight.sram_extra = 2*seq*word
        kernel.gemm_input1_weight.dram_extra = 2*seq*word
        
    for i in range(stage-1):
        connection = dataflow_graph.connections.add()
        connection.id = counter+i
        connection.startIdx = counter+i
        connection.endIdx = counter+i+1
        
        




dataflow_graph = setup_pb2.Dataflow_Graph()

kernel = dataflow_graph.kernels.add()
kernel.name = "Q"
kernel.id = 0
kernel.fwd_bwd = 1
kernel.type = 1
kernel.config = -1
kernel.gemm_input1_weight.outer = num_head
kernel.gemm_input1_weight.M = head_dim
kernel.gemm_input1_weight.K = hidden
kernel.gemm_input1_weight.N = seq
kernel.gemm_input1_weight.input_tensor_size = hidden*seq*word
kernel.gemm_input1_weight.weight_tensor_size = hidden*hidden*word
kernel.gemm_input1_weight.output_tensor_size = hidden*seq*word
kernel.gemm_input1_weight.tiling = 4


kernel = dataflow_graph.kernels.add()
kernel.name = "K"
kernel.id = 1
kernel.fwd_bwd = 1
kernel.type = 1
kernel.config = -1
kernel.gemm_input1_weight.outer = num_head
kernel.gemm_input1_weight.M = head_dim
kernel.gemm_input1_weight.K = hidden
kernel.gemm_input1_weight.N = seq
kernel.gemm_input1_weight.input_tensor_size = hidden*seq*word
kernel.gemm_input1_weight.weight_tensor_size = hidden*hidden*word
kernel.gemm_input1_weight.output_tensor_size = hidden*seq*word
kernel.gemm_input1_weight.tiling = 4


kernel = dataflow_graph.kernels.add()
kernel.name = "V"
kernel.id = 2
kernel.fwd_bwd = 1
kernel.type = 1
kernel.config = -1
kernel.gemm_input1_weight.outer = num_head
kernel.gemm_input1_weight.M = head_dim
kernel.gemm_input1_weight.K = hidden
kernel.gemm_input1_weight.N = seq
kernel.gemm_input1_weight.input_tensor_size = hidden*seq*word
kernel.gemm_input1_weight.weight_tensor_size = hidden*hidden*word
kernel.gemm_input1_weight.output_tensor_size = hidden*seq*word
kernel.gemm_input1_weight.tiling = 4








# six FFT
FFT(0+3, 'Q')
FFT(1*stage+3, 'K')
FFT(2*stage+3, 'QKi')
FFT(3*stage+3, 'inter')
FFT(4*stage+3, 'V')
FFT(5*stage+3, 'interVi')

# QK multiply
kernel = dataflow_graph.kernels.add()
kernel.name = 'QKmultiply'
kernel.id = 6*stage+3
kernel.fwd_bwd = 1
kernel.type = 2
kernel.config = -1
kernel.elementwise_input1_input2.outer = 1
kernel.elementwise_input1_input2.M = 2*seq
kernel.elementwise_input1_input2.N = 1
kernel.elementwise_input1_input2.input_tensor_1_size = 2*seq*word
kernel.elementwise_input1_input2.input_tensor_2_size = 2*seq*word
kernel.elementwise_input1_input2.output_tensor_size = 2*seq*word
kernel.elementwise_input1_input2.tiling = 5
kernel.elementwise_input1_input2.num_input = hidden

# softmax
kernel = dataflow_graph.kernels.add()
kernel.name = 'softmax'
kernel.id = 6*stage+1+3
kernel.fwd_bwd = 1
kernel.type = 2
kernel.config = -1
kernel.elementwise_input1_input2.outer = 1
kernel.elementwise_input1_input2.M = 2*seq
kernel.elementwise_input1_input2.N = 1
kernel.elementwise_input1_input2.input_tensor_1_size = 2*seq*word
kernel.elementwise_input1_input2.input_tensor_2_size = 2*seq*word
kernel.elementwise_input1_input2.output_tensor_size = 2*seq*word
kernel.elementwise_input1_input2.tiling = 5
kernel.elementwise_input1_input2.num_input = hidden

# interV multiply
kernel = dataflow_graph.kernels.add()
kernel.name = 'interVmultiply'
kernel.id = 6*stage+2+3
kernel.fwd_bwd = 1
kernel.type = 2
kernel.config = -1
kernel.elementwise_input1_input2.outer = 1
kernel.elementwise_input1_input2.M = 2*seq
kernel.elementwise_input1_input2.N = 1
kernel.elementwise_input1_input2.input_tensor_1_size = 2*seq*word
kernel.elementwise_input1_input2.input_tensor_2_size = 2*seq*word
kernel.elementwise_input1_input2.output_tensor_size = 2*seq*word
kernel.elementwise_input1_input2.tiling = 5
kernel.elementwise_input1_input2.num_input = hidden



kernel = dataflow_graph.kernels.add()
kernel.name = "FFN0"
kernel.id = 6*stage+3+3
kernel.fwd_bwd = 1
kernel.type = 1
kernel.config = -1
kernel.gemm_input1_weight.outer = 1
kernel.gemm_input1_weight.M = mlp_dim
kernel.gemm_input1_weight.K = hidden
kernel.gemm_input1_weight.N = seq
kernel.gemm_input1_weight.input_tensor_size = hidden*seq*word
kernel.gemm_input1_weight.weight_tensor_size = mlp_dim*hidden*word
kernel.gemm_input1_weight.output_tensor_size = mlp_dim*seq*word
kernel.gemm_input1_weight.tiling = 4

kernel = dataflow_graph.kernels.add()
kernel.name = "FFN1"
kernel.id = 6*stage+4+3
kernel.fwd_bwd = 1
kernel.type = 1
kernel.config = -1 
kernel.gemm_input1_weight.outer = 1
kernel.gemm_input1_weight.M = hidden
kernel.gemm_input1_weight.K = mlp_dim
kernel.gemm_input1_weight.N = seq
kernel.gemm_input1_weight.input_tensor_size = mlp_dim*seq*word
kernel.gemm_input1_weight.weight_tensor_size = mlp_dim*hidden*word
kernel.gemm_input1_weight.output_tensor_size = hidden*seq*word
kernel.gemm_input1_weight.tiling = 4













connection = dataflow_graph.connections.add()
connection.id = 10*stage
connection.startIdx = stage-1+3
connection.endIdx = 6*stage+3

connection = dataflow_graph.connections.add()
connection.id = 10*stage+1
connection.startIdx = 2*stage-1+3
connection.endIdx = 6*stage+3

connection = dataflow_graph.connections.add()
connection.id = 10*stage+2
connection.startIdx = 6*stage+3
connection.endIdx = 2*stage+3




connection = dataflow_graph.connections.add()
connection.id = 10*stage+3
connection.startIdx = 3*stage-1+3
connection.endIdx = 6*stage+1+3

connection = dataflow_graph.connections.add()
connection.id = 10*stage+4
connection.startIdx = 6*stage+1+3
connection.endIdx = 3*stage+3




connection = dataflow_graph.connections.add()
connection.id = 10*stage+5
connection.startIdx = 4*stage-1+3
connection.endIdx = 6*stage+2+3

connection = dataflow_graph.connections.add()
connection.id = 10*stage+6
connection.startIdx = 5*stage-1+3
connection.endIdx = 6*stage+2+3

connection = dataflow_graph.connections.add()
connection.id = 10*stage+7
connection.startIdx = 6*stage+2+3
connection.endIdx = 5*stage+3





connection = dataflow_graph.connections.add()
connection.id = 10*stage+8
connection.startIdx = 6*stage+2
connection.endIdx = 6*stage+3+3

connection = dataflow_graph.connections.add()
connection.id = 10*stage+9
connection.startIdx = 6*stage+3+3
connection.endIdx = 6*stage+4+3






connection = dataflow_graph.connections.add()
connection.id = 10*stage+10
connection.startIdx = 0
connection.endIdx = 0+3

connection = dataflow_graph.connections.add()
connection.id = 10*stage+11
connection.startIdx = 1
connection.endIdx = 1*stage+3

connection = dataflow_graph.connections.add()
connection.id = 10*stage+12
connection.startIdx = 2
connection.endIdx = 4*stage+3


# write to text file
with open('generator/Regular_FFT_Hyena.txt', "w") as file:
    text_format.PrintMessage(dataflow_graph, file)
    
    