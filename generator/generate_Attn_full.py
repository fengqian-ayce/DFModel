import setup_pb2
import csv
import argparse
from google.protobuf import text_format
import pydot

# user pass in
parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, required=True)
parser.add_argument('--seq', type=int, required=True)
parser.add_argument('--num_head', type=int, required=True)
parser.add_argument('--head_dim', type=int, required=True)
parser.add_argument('--word', type=int, required=True)
parser.add_argument('--mlp_dim', type=int, required=True)
args = parser.parse_args()


hidden = args.hidden
seq = args.seq
num_head = args.num_head
head_dim = args.head_dim
word = args.word
mlp_dim = args.mlp_dim

if num_head * head_dim != hidden:
    raise Exception('Wrong!')



dataflow_graph = setup_pb2.Dataflow_Graph()

for i in range(1, 9):
    kernel = dataflow_graph.kernels.add()
    
    if i == 1:
        kernel.name = "Q"
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

    elif i == 2:
        kernel.name = "K"
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

    elif i == 3:
        kernel.name = "V"
        kernel.id = 3
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

    elif i == 4:
        kernel.name = "MHA_GEMM_1"
        kernel.id = 4
        kernel.fwd_bwd = 1
        kernel.type = 1
        kernel.config = -1
        kernel.gemm_input1_input2.outer = num_head
        kernel.gemm_input1_input2.M = seq
        kernel.gemm_input1_input2.K = head_dim
        kernel.gemm_input1_input2.N = seq
        kernel.gemm_input1_input2.input_tensor_1_size = hidden*seq*word
        kernel.gemm_input1_input2.input_tensor_2_size = hidden*seq*word
        kernel.gemm_input1_input2.output_tensor_size = seq*seq*num_head*word
        kernel.gemm_input1_input2.tiling = 4
        
    elif i == 5:
        kernel.name = "SOFTMAX"
        kernel.id = 5
        kernel.fwd_bwd = 1
        kernel.type = 2
        kernel.config = -1
        kernel.elementwise_input1.outer = num_head
        kernel.elementwise_input1.M = seq
        kernel.elementwise_input1.N = seq
        kernel.elementwise_input1.input_tensor_size = seq*seq*num_head*word
        kernel.elementwise_input1.output_tensor_size = seq*seq*num_head*word
        kernel.elementwise_input1.tiling = 4

    elif i == 6:
        kernel.name = "MHA_GEMM_2"
        kernel.id = 6
        kernel.fwd_bwd = 1
        kernel.type = 1
        kernel.config = -1
        kernel.gemm_input1_input2.outer = num_head
        kernel.gemm_input1_input2.M = head_dim
        kernel.gemm_input1_input2.K = seq
        kernel.gemm_input1_input2.N = seq
        kernel.gemm_input1_input2.input_tensor_1_size = hidden*seq*word
        kernel.gemm_input1_input2.input_tensor_2_size = seq*seq*num_head*word
        kernel.gemm_input1_input2.output_tensor_size = hidden*seq*word
        kernel.gemm_input1_input2.tiling = 3

    elif i == 7:
        kernel.name = "FFN0"
        kernel.id = 7
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

    elif i == 8:
        kernel.name = "FFN1"
        kernel.id = 8
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

for i in range(1, 8):
    connection = dataflow_graph.connections.add()
    
    if i == 1:
        connection.id = i
        connection.startIdx = 1
        connection.endIdx = 4
    
    elif i == 2:
        connection.id = i
        connection.startIdx = 2
        connection.endIdx = 4
    
    elif i == 3:
        connection.id = i
        connection.startIdx = 4
        connection.endIdx = 5
    
    elif i == 4:
        connection.id = i
        connection.startIdx = 5
        connection.endIdx = 6
    
    elif i == 5:
        connection.id = i
        connection.startIdx = 3
        connection.endIdx = 6

    elif i == 6:
        connection.id = i
        connection.startIdx = 6
        connection.endIdx = 7

    elif i == 7:
        connection.id = i
        connection.startIdx = 7
        connection.endIdx = 8


# write to text file
with open('generator/Attn_full.txt', "w") as file:
    text_format.PrintMessage(dataflow_graph, file)