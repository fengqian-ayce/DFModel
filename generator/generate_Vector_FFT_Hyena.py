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


# total 6*stage+3 kernels
stage = int(math.log2(seq) / math.log2(r))

def FFT(counter, name):
    for i in range(stage):
        kernel = dataflow_graph.kernels.add()
        kernel.name = name+'_stage'+str(i)
        kernel.id = counter+i
        kernel.fwd_bwd = 1
        kernel.type = 1
        kernel.config = -1
        kernel.gemm_input1_weight.outer = 1
        kernel.gemm_input1_weight.M = 2*r*int(seq/r)
        kernel.gemm_input1_weight.K = 2*int(math.log2(r))
        kernel.gemm_input1_weight.N = 1
        kernel.gemm_input1_weight.input_tensor_size = 2*seq*word
        kernel.gemm_input1_weight.output_tensor_size = 2*seq*word
        kernel.gemm_input1_weight.tiling = 5
        kernel.gemm_input1_weight.skip_weight = True
        kernel.gemm_input1_weight.use_effective_stage = True
        kernel.gemm_input1_weight.num_input = hidden

    for i in range(stage-1):
        connection = dataflow_graph.connections.add()
        connection.id = counter+i
        connection.startIdx = counter+i
        connection.endIdx = counter+i+1
        
        




dataflow_graph = setup_pb2.Dataflow_Graph()

FFT(0, 'Q')
FFT(1*stage, 'K')
FFT(2*stage, 'QKi')
FFT(3*stage, 'inter')
FFT(4*stage, 'V')
FFT(5*stage, 'interVi')





# QK multiply
kernel = dataflow_graph.kernels.add()
kernel.name = 'QKmultiply'
kernel.id = 6*stage
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
kernel.id = 6*stage+1
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
kernel.id = 6*stage+2
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




connection = dataflow_graph.connections.add()
connection.id = 10*stage
connection.startIdx = stage-1
connection.endIdx = 6*stage

connection = dataflow_graph.connections.add()
connection.id = 10*stage+1
connection.startIdx = 2*stage-1
connection.endIdx = 6*stage

connection = dataflow_graph.connections.add()
connection.id = 10*stage+2
connection.startIdx = 6*stage
connection.endIdx = 2*stage




connection = dataflow_graph.connections.add()
connection.id = 10*stage+3
connection.startIdx = 3*stage-1
connection.endIdx = 6*stage+1

connection = dataflow_graph.connections.add()
connection.id = 10*stage+4
connection.startIdx = 6*stage+1
connection.endIdx = 3*stage




connection = dataflow_graph.connections.add()
connection.id = 10*stage+5
connection.startIdx = 4*stage-1
connection.endIdx = 6*stage+2

connection = dataflow_graph.connections.add()
connection.id = 10*stage+6
connection.startIdx = 5*stage-1
connection.endIdx = 6*stage+2

connection = dataflow_graph.connections.add()
connection.id = 10*stage+7
connection.startIdx = 6*stage+2
connection.endIdx = 5*stage




# write to text file
with open('generator/Vector_FFT_Hyena.txt', "w") as file:
    text_format.PrintMessage(dataflow_graph, file)