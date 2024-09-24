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
parser.add_argument('--layer', type=int, required=True)
parser.add_argument('--mlp_dim', type=int, required=True)
args = parser.parse_args()


hidden = args.hidden
seq = args.seq
num_head = args.num_head
head_dim = args.head_dim
word = args.word
layer = args.layer
mlp_dim = args.mlp_dim


if num_head * head_dim != hidden:
    raise Exception('Wrong!')


dataflow_graph = setup_pb2.Dataflow_Graph()

for l in range(0, layer):
    for i in range(1, 19):
        kernel = dataflow_graph.kernels.add()
        
        if i == 1:
            kernel.name = "Add_Prev_Layer"+"_layer_"+str(l)
            kernel.id = 1+18*l
            kernel.fwd_bwd = 1
            kernel.type = 2
            kernel.config = -1
            kernel.elementwise_input1.outer = 1
            kernel.elementwise_input1.M = hidden
            kernel.elementwise_input1.N = seq		
            kernel.elementwise_input1.input_tensor_size = hidden*seq*word
            kernel.elementwise_input1.output_tensor_size = hidden*seq*word
            kernel.elementwise_input1.tiling = 4
            
        elif i == 2:
            kernel.name = "LayerNorm_1"+"_layer_"+str(l)
            kernel.id = 2+18*l
            kernel.fwd_bwd = 1
            kernel.type = 2
            kernel.config = -1  
            kernel.elementwise_input1.outer = 1
            kernel.elementwise_input1.M = hidden
            kernel.elementwise_input1.N = seq  
            kernel.elementwise_input1.input_tensor_size = hidden*seq*word
            kernel.elementwise_input1.output_tensor_size = hidden*seq*word
            kernel.elementwise_input1.tiling = 4
              
        elif i == 3:
            kernel.name = "Q"+"_layer_"+str(l)
            kernel.id = 3+18*l
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
            kernel.name = "K"+"_layer_"+str(l)
            kernel.id = 4+18*l
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

        elif i == 5:
            kernel.name = "V"+"_layer_"+str(l)
            kernel.id = 5+18*l
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

        elif i == 6:
            kernel.name = "MHA_GEMM_1"+"_layer_"+str(l)
            kernel.id = 6+18*l
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
            
        elif i == 7:
            kernel.name = "SOFTMAX"+"_layer_"+str(l)
            kernel.id = 7+18*l
            kernel.fwd_bwd = 1
            kernel.type = 2
            kernel.config = -1
            kernel.elementwise_input1.outer = num_head
            kernel.elementwise_input1.M = seq
            kernel.elementwise_input1.N = seq
            kernel.elementwise_input1.input_tensor_size = seq*seq*num_head*word
            kernel.elementwise_input1.output_tensor_size = seq*seq*num_head*word
            kernel.elementwise_input1.tiling = 4

        elif i == 8:
            kernel.name = "DropOut_1"+"_layer_"+str(l)
            kernel.id = 8+18*l
            kernel.fwd_bwd = 1
            kernel.type = 2
            kernel.config = -1
            kernel.elementwise_input1.outer = num_head
            kernel.elementwise_input1.M = seq
            kernel.elementwise_input1.N = seq
            kernel.elementwise_input1.input_tensor_size = seq*seq*num_head*word
            kernel.elementwise_input1.output_tensor_size = seq*seq*num_head*word
            kernel.elementwise_input1.tiling = 4

        elif i == 9:
            kernel.name = "MHA_GEMM_2"+"_layer_"+str(l)
            kernel.id = 9+18*l
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
            kernel.gemm_input1_input2.tiling = 4

        elif i == 10:
            kernel.name = "PROJ_GEMM"+"_layer_"+str(l)
            kernel.id = 10+18*l
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

        elif i == 11:
            kernel.name = "DropOut_2"+"_layer_"+str(l)
            kernel.id = 11+18*l
            kernel.fwd_bwd = 1
            kernel.type = 2
            kernel.config = -1
            kernel.elementwise_input1.outer = 1
            kernel.elementwise_input1.M = hidden
            kernel.elementwise_input1.N = seq
            kernel.elementwise_input1.input_tensor_size = hidden*seq*word
            kernel.elementwise_input1.output_tensor_size = hidden*seq*word
            kernel.elementwise_input1.tiling = 4

        elif i == 12:
            kernel.name = "Add_1"+"_layer_"+str(l)
            kernel.id = 12+18*l
            kernel.fwd_bwd = 1
            kernel.type = 2
            kernel.config = -1
            kernel.elementwise_input1_input2.outer = 1
            kernel.elementwise_input1_input2.M = hidden
            kernel.elementwise_input1_input2.N = seq  
            kernel.elementwise_input1_input2.input_tensor_1_size = hidden*seq*word
            kernel.elementwise_input1_input2.input_tensor_2_size = hidden*seq*word
            kernel.elementwise_input1_input2.output_tensor_size = hidden*seq*word
            kernel.elementwise_input1_input2.tiling = 4

        elif i == 13:
            kernel.name = "LayerNorm_2"+"_layer_"+str(l)
            kernel.id = 13+18*l
            kernel.fwd_bwd = 1
            kernel.type = 2
            kernel.config = -1  
            kernel.elementwise_input1.outer = 1
            kernel.elementwise_input1.M = hidden
            kernel.elementwise_input1.N = seq
            kernel.elementwise_input1.input_tensor_size = hidden*seq*word
            kernel.elementwise_input1.output_tensor_size = hidden*seq*word
            kernel.elementwise_input1.tiling = 4

        elif i == 14:
            kernel.name = "FFN0"+"_layer_"+str(l)
            kernel.id = 14+18*l
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
        
        elif i == 15:
            kernel.name = "GeLU"+"_layer_"+str(l)
            kernel.id = 15+18*l
            kernel.fwd_bwd = 1
            kernel.type = 2
            kernel.config = -1
            kernel.elementwise_input1.outer = 1
            kernel.elementwise_input1.M = mlp_dim
            kernel.elementwise_input1.N = seq 
            kernel.elementwise_input1.input_tensor_size = mlp_dim*seq*word
            kernel.elementwise_input1.output_tensor_size = mlp_dim*seq*word
            kernel.elementwise_input1.tiling = 4

        elif i == 16:
            kernel.name = "FFN1"+"_layer_"+str(l)
            kernel.id = 16+18*l
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

        elif i == 17:
            kernel.name = "DropOut_3"+"_layer_"+str(l)
            kernel.id = 17+18*l
            kernel.fwd_bwd = 1
            kernel.type = 2
            kernel.config = -1
            kernel.elementwise_input1.outer = 1
            kernel.elementwise_input1.M = hidden
            kernel.elementwise_input1.N = seq
            kernel.elementwise_input1.input_tensor_size = hidden*seq*word
            kernel.elementwise_input1.output_tensor_size = hidden*seq*word
            kernel.elementwise_input1.tiling = 4

        elif i == 18:
            kernel.name = "Add_2"+"_layer_"+str(l)
            kernel.id = 18+18*l
            kernel.fwd_bwd = 1
            kernel.type = 2
            kernel.config = -1
            kernel.elementwise_input1_input2.outer = 1
            kernel.elementwise_input1_input2.M = hidden
            kernel.elementwise_input1_input2.N = seq
            kernel.elementwise_input1_input2.input_tensor_1_size = hidden*seq*word
            kernel.elementwise_input1_input2.input_tensor_2_size = hidden*seq*word
            kernel.elementwise_input1_input2.output_tensor_size = hidden*seq*word
            kernel.elementwise_input1_input2.tiling = 4


    for i in range(1, 22):
        connection = dataflow_graph.connections.add()
        
        if i == 1:
            connection.id = i+53*l
            connection.startIdx = 1+18*l
            connection.endIdx = 2+18*l
        
        elif i == 2:
            connection.id = i+53*l
            connection.startIdx = 2+18*l
            connection.endIdx = 3+18*l
        
        elif i == 3:
            connection.id = i+53*l
            connection.startIdx = 2+18*l
            connection.endIdx = 4+18*l
        
        elif i == 4:
            connection.id = i+53*l
            connection.startIdx = 2+18*l
            connection.endIdx = 5+18*l
        
        elif i == 5:
            connection.id = i+53*l
            connection.startIdx = 3+18*l
            connection.endIdx = 6+18*l
        
        elif i == 6:
            connection.id = i+53*l
            connection.startIdx = 4+18*l
            connection.endIdx = 6+18*l

        elif i == 7:
            connection.id = i+53*l
            connection.startIdx = 6+18*l
            connection.endIdx = 7+18*l
        
        elif i == 8:
            connection.id = i+53*l
            connection.startIdx = 7+18*l
            connection.endIdx = 8+18*l
        
        elif i == 9:
            connection.id = i+53*l
            connection.startIdx = 5+18*l
            connection.endIdx = 9+18*l
            
        elif i == 10:
            connection.id = i+53*l
            connection.startIdx = 8+18*l
            connection.endIdx = 9+18*l
        
        elif i == 11:
            connection.id = i+53*l
            connection.startIdx = 9+18*l
            connection.endIdx = 10+18*l
        
        elif i == 12:
            connection.id = i+53*l
            connection.startIdx = 10+18*l
            connection.endIdx = 11+18*l
        
        elif i == 13:
            connection.id = i+53*l
            connection.startIdx = 11+18*l
            connection.endIdx = 12+18*l
        
        elif i == 14:
            connection.id = i+53*l
            connection.startIdx = 1+18*l
            connection.endIdx = 12+18*l
        
        elif i == 15:
            connection.id = i+53*l
            connection.startIdx = 12+18*l
            connection.endIdx = 13+18*l
        
        elif i == 16:
            connection.id = i+53*l
            connection.startIdx = 13+18*l
            connection.endIdx = 14+18*l
        
        elif i == 17:
            connection.id = i+53*l
            connection.startIdx = 14+18*l
            connection.endIdx = 15+18*l
        
        elif i == 18:
            connection.id = i+53*l
            connection.startIdx = 15+18*l
            connection.endIdx = 16+18*l
            
        elif i == 19:
            connection.id = i+53*l
            connection.startIdx = 16+18*l
            connection.endIdx = 17+18*l
            
        elif i == 20:
            connection.id = i+53*l
            connection.startIdx = 17+18*l
            connection.endIdx = 18+18*l
        
        elif i == 21:
            connection.id = i+53*l
            connection.startIdx = 12+18*l
            connection.endIdx = 18+18*l
            
            
        
for l in range(1, layer):
    connection = dataflow_graph.connections.add()
    
    connection.id = 21*layer+l
    connection.startIdx = 18*l
    connection.endIdx = 18*l+1
       
# write to text file
with open('generator/LLM_fwd_multilayer.txt', "w") as file:
    text_format.PrintMessage(dataflow_graph, file)