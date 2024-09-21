import setup_pb2
import csv
import argparse
from google.protobuf import text_format
import pydot
import math



# user pass in
parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, required=True)
parser.add_argument('--B', type=int, required=True)
parser.add_argument('--iteration', type=int, required=True)
parser.add_argument('--word', type=int, required=True)
args = parser.parse_args()


N = args.N
B = args.B
iteration = args.iteration
word = args.word


        #Y
    ###########
#X  ###########
    ###########
     
    
dataflow_graph = setup_pb2.Dataflow_Graph()

for i in range(iteration):
    side = N - i*B
    
    kernel = dataflow_graph.kernels.add()
    kernel.name = "Iteration_"+str(i+1)
    kernel.id = i+1
    kernel.type = 1
    kernel.gemm_input1_input2.outer = 1
    kernel.gemm_input1_input2.M = side
    kernel.gemm_input1_input2.K = side
    kernel.gemm_input1_input2.N = B
    kernel.gemm_input1_input2.input_tensor_1_size = side*B*word
    kernel.gemm_input1_input2.output_tensor_size = side*B*word
    kernel.gemm_input1_input2.tiling = 5

    kernel.gemm_input1_input2.communication_type = 6
    kernel.gemm_input1_input2.communication_size = side*B*word

    kernel.gemm_input1_input2.communication_type_2 = 5
    kernel.gemm_input1_input2.communication_size_2 = side*B*word
        
    
    
for i in range(iteration-1):
    connection = dataflow_graph.connections.add()
    connection.id = i+1
    connection.startIdx = i+1
    connection.endIdx = i+2
    
    
# write to text file
with open('generator/HPL.txt', "w") as file:
    text_format.PrintMessage(dataflow_graph, file)