import setup_pb2
import csv
import argparse
from google.protobuf import text_format
import pydot
import math



# user pass in
parser = argparse.ArgumentParser()
parser.add_argument('--length', type=int, required=True)
parser.add_argument('--r', type=int, required=True)
parser.add_argument('--word', type=int, required=True)
args = parser.parse_args()


length = args.length
r = args.r
word = args.word


N = int(length**0.5)

    
dataflow_graph = setup_pb2.Dataflow_Graph()

# step 1
kernel = dataflow_graph.kernels.add()
kernel.name = "Step_1"
kernel.id = 1
kernel.type = 1
kernel.gemm_input1_weight.outer = 1
kernel.gemm_input1_weight.M = N
kernel.gemm_input1_weight.K = int(4*math.log2(N)/math.log2(r)*r)
kernel.gemm_input1_weight.N = N
kernel.gemm_input1_weight.input_tensor_size = N*N*word
kernel.gemm_input1_weight.output_tensor_size = N*N*word
kernel.gemm_input1_weight.tiling = 5


# step 2
kernel = dataflow_graph.kernels.add()
kernel.name = "Step_2"
kernel.id = 2
kernel.type = 1
kernel.gemm_input1_weight.outer = 1
kernel.gemm_input1_weight.M = N
kernel.gemm_input1_weight.K = int(4*math.log2(N)/math.log2(r)*r)
kernel.gemm_input1_weight.N = N
kernel.gemm_input1_weight.input_tensor_size = N*N*word
kernel.gemm_input1_weight.output_tensor_size = N*N*word
kernel.gemm_input1_weight.tiling = 5
kernel.gemm_input1_weight.communication_type = 2
kernel.gemm_input1_weight.communication_size = N*N*word



   
# first connection
connection = dataflow_graph.connections.add()
connection.id = 1
connection.startIdx = 1
connection.endIdx = 2

    
    
# write to text file
with open('generator/FFT.txt', "w") as file:
    text_format.PrintMessage(dataflow_graph, file)
    