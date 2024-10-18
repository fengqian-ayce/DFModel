import setup_pb2
import csv
import argparse
from google.protobuf import text_format
import pydot
import math


# user pass in
parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, required=True)
parser.add_argument('--seq', type=int, required=True)
parser.add_argument('--word', type=int, required=True)
parser.add_argument('--r', type=int, required=True)
args = parser.parse_args()


hidden = args.hidden
seq = args.seq
word = args.word
r = args.r


stage = math.ceil(math.log2(seq) / math.log2(r))

dataflow_graph = setup_pb2.Dataflow_Graph()


cnt = 1

# proj1
kernel = dataflow_graph.kernels.add()
kernel.name = 'Proj1'
kernel.id = cnt
kernel.fwd_bwd = 1
kernel.type = 1
kernel.config = -1
kernel.gemm_input1_weight.outer = 1
kernel.gemm_input1_weight.M = hidden
kernel.gemm_input1_weight.K = hidden
kernel.gemm_input1_weight.N = seq
kernel.gemm_input1_weight.input_tensor_size = hidden*seq*word
kernel.gemm_input1_weight.weight_tensor_size = hidden*hidden*word
kernel.gemm_input1_weight.output_tensor_size = hidden*seq*word
kernel.gemm_input1_weight.tiling = 4
cnt += 1


# proj2
kernel = dataflow_graph.kernels.add()
kernel.name = 'Proj2'
kernel.id = cnt
kernel.fwd_bwd = 1
kernel.type = 1
kernel.config = -1
kernel.gemm_input1_weight.outer = 1
kernel.gemm_input1_weight.M = hidden
kernel.gemm_input1_weight.K = hidden
kernel.gemm_input1_weight.N = seq
kernel.gemm_input1_weight.input_tensor_size = hidden*seq*word
kernel.gemm_input1_weight.weight_tensor_size = hidden*hidden*word
kernel.gemm_input1_weight.output_tensor_size = hidden*seq*word
kernel.gemm_input1_weight.tiling = 4
cnt += 1


# conv
kernel = dataflow_graph.kernels.add()
kernel.name = "Conv"
kernel.id = cnt
kernel.fwd_bwd = 1
kernel.type = 1
kernel.config = -1
kernel.gemm_input1_weight.outer = 1
kernel.gemm_input1_weight.M = seq
kernel.gemm_input1_weight.K = hidden
kernel.gemm_input1_weight.N = 1
kernel.gemm_input1_weight.input_tensor_size = seq*word
kernel.gemm_input1_weight.output_tensor_size = seq*word
kernel.gemm_input1_weight.tiling = 2
kernel.gemm_input1_weight.skip_weight = True
cnt += 1


for i in range(stage):
    kernel = dataflow_graph.kernels.add()
    kernel.name = 'Scan_stage_'+str(i)
    kernel.id = cnt
    kernel.fwd_bwd = 1
    kernel.type = 1
    kernel.config = -1
    kernel.gemm_input1_weight.outer = 1
    kernel.gemm_input1_weight.M = seq
    kernel.gemm_input1_weight.K = int(math.log2(r))
    kernel.gemm_input1_weight.N = 1
    kernel.gemm_input1_weight.input_tensor_size = seq*word
    kernel.gemm_input1_weight.output_tensor_size = seq*word
    kernel.gemm_input1_weight.tiling = 5
    kernel.gemm_input1_weight.skip_weight = True
    kernel.gemm_input1_weight.use_effective_stage = True
    kernel.gemm_input1_weight.num_input = hidden
    cnt += 1





# multiply
kernel = dataflow_graph.kernels.add()
kernel.name = "Multiply"
kernel.id = cnt
kernel.fwd_bwd = 1
kernel.type = 2
kernel.config = -1
kernel.elementwise_input1_input2.outer = 1
kernel.elementwise_input1_input2.M = seq
kernel.elementwise_input1_input2.N = hidden
kernel.elementwise_input1_input2.input_tensor_1_size = hidden*seq*word
kernel.elementwise_input1_input2.input_tensor_2_size = hidden*seq*word
kernel.elementwise_input1_input2.output_tensor_size = hidden*seq*word
kernel.elementwise_input1_input2.tiling = 2
cnt += 1


# proj 3
kernel = dataflow_graph.kernels.add()
kernel.name = 'Proj3'
kernel.id = cnt
kernel.fwd_bwd = 1
kernel.type = 1
kernel.config = -1
kernel.gemm_input1_weight.outer = 1
kernel.gemm_input1_weight.M = hidden
kernel.gemm_input1_weight.K = hidden
kernel.gemm_input1_weight.N = seq
kernel.gemm_input1_weight.input_tensor_size = hidden*seq*word
kernel.gemm_input1_weight.weight_tensor_size = hidden*hidden*word
kernel.gemm_input1_weight.output_tensor_size = hidden*seq*word
kernel.gemm_input1_weight.tiling = 4






cnt = 0

connection = dataflow_graph.connections.add()
connection.id = cnt
connection.startIdx = 1
connection.endIdx = 3
cnt += 1

connection = dataflow_graph.connections.add()
connection.id = cnt
connection.startIdx = 3
connection.endIdx = 4
cnt += 1

for i in range(4, 4+stage):
    connection = dataflow_graph.connections.add()
    connection.id = cnt
    connection.startIdx = i
    connection.endIdx = i+1
    cnt += 1

connection = dataflow_graph.connections.add()
connection.id = cnt
connection.startIdx = 2
connection.endIdx = 3+stage+1
cnt += 1

connection = dataflow_graph.connections.add()
connection.id = cnt
connection.startIdx = 3+stage+1
connection.endIdx = 3+stage+2




# write to text file
with open('generator/Mamba_w_hsscan_hw.txt', "w") as file:
    text_format.PrintMessage(dataflow_graph, file)
    