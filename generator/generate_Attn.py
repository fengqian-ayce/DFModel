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
args = parser.parse_args()


hidden = args.hidden
seq = args.seq
num_head = args.num_head
head_dim = args.head_dim
word = args.word


if num_head * head_dim != hidden:
    raise Exception('Wrong!')



dataflow_graph = setup_pb2.Dataflow_Graph()

for i in range(1, 4):
    kernel = dataflow_graph.kernels.add()

    if i == 1:
        kernel.name = "MHA_GEMM_1"
        kernel.id = 1
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
        
    elif i == 2:
        kernel.name = "SOFTMAX"
        kernel.id = 2
        kernel.fwd_bwd = 1
        kernel.type = 2
        kernel.config = -1
        kernel.elementwise_input1.outer = num_head
        kernel.elementwise_input1.M = seq
        kernel.elementwise_input1.N = seq
        kernel.elementwise_input1.input_tensor_size = seq*seq*num_head*word
        kernel.elementwise_input1.output_tensor_size = seq*seq*num_head*word
        kernel.elementwise_input1.tiling = 4

    elif i == 3:
        kernel.name = "MHA_GEMM_2"
        kernel.id = 3
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
          

for i in range(1, 3):
    connection = dataflow_graph.connections.add()
    
    if i == 1:
        connection.id = 1
        connection.startIdx = 1
        connection.endIdx = 2
    
    elif i == 2:
        connection.id = 2
        connection.startIdx = 2
        connection.endIdx = 3

# write to text file
with open('generator/Attn.txt', "w") as file:
    text_format.PrintMessage(dataflow_graph, file)