import setup_pb2
import csv
import argparse
from google.protobuf import text_format
import pydot

# user pass in
parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, required=True)
parser.add_argument('--num_head', type=int, required=True)
parser.add_argument('--head_dim', type=int, required=True)
parser.add_argument('--word', type=int, required=True)
parser.add_argument('--seq_window', type=int, required=True)
parser.add_argument('--tp', type=int, required=True)
parser.add_argument('--mlp_dim', type=int, required=True)
args = parser.parse_args()


hidden = args.hidden
num_head = args.num_head
head_dim = args.head_dim
word = args.word
seq_window = args.seq_window
tp = args.tp
mlp_dim = args.mlp_dim

seq = 1


if num_head * head_dim != hidden:
    raise Exception('Wrong!')



dataflow_graph = setup_pb2.Dataflow_Graph()

for i in range(1, 21):
    kernel = dataflow_graph.kernels.add()
    
    if i == 1:
        kernel.name = "Add_Prev_Layer"
        kernel.id = 1
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
        kernel.name = "LayerNorm_1"
        kernel.id = 2
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
        kernel.name = "Q"
        kernel.id = 3
        kernel.fwd_bwd = 1
        kernel.type = 1
        kernel.config = -1
        kernel.gemm_input1_weight.memory_size = hidden*hidden*word/tp
        kernel.gemm_input1_weight.outer = num_head
        kernel.gemm_input1_weight.M = head_dim
        kernel.gemm_input1_weight.K = hidden
        kernel.gemm_input1_weight.N = seq
        kernel.gemm_input1_weight.input_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.weight_tensor_size = hidden*hidden*word
        kernel.gemm_input1_weight.output_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.tiling = 4

    elif i == 4:
        kernel.name = "K"
        kernel.id = 4
        kernel.fwd_bwd = 1
        kernel.type = 1
        kernel.config = -1
        kernel.gemm_input1_weight.memory_size = hidden*hidden*word/tp
        kernel.gemm_input1_weight.outer = num_head
        kernel.gemm_input1_weight.M = head_dim
        kernel.gemm_input1_weight.K = hidden
        kernel.gemm_input1_weight.N = seq
        kernel.gemm_input1_weight.input_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.weight_tensor_size = hidden*hidden*word
        kernel.gemm_input1_weight.output_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.tiling = 4

    elif i == 5:
        kernel.name = "V"
        kernel.id = 5
        kernel.fwd_bwd = 1
        kernel.type = 1
        kernel.config = -1
        kernel.gemm_input1_weight.memory_size = hidden*hidden*word/tp
        kernel.gemm_input1_weight.outer = num_head
        kernel.gemm_input1_weight.M = head_dim
        kernel.gemm_input1_weight.K = hidden
        kernel.gemm_input1_weight.N = seq
        kernel.gemm_input1_weight.input_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.weight_tensor_size = hidden*hidden*word
        kernel.gemm_input1_weight.output_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.tiling = 4

    elif i == 6:
        kernel.name = "K_cache"
        kernel.id = 6
        kernel.fwd_bwd = 1
        kernel.type = 1
        kernel.config = -1
        kernel.elementwise_input1.memory_size = word*hidden*seq_window/tp
        kernel.elementwise_input1.outer = num_head
        kernel.elementwise_input1.M = head_dim
        kernel.elementwise_input1.N = seq_window+1
        kernel.elementwise_input1.input_tensor_size = hidden*seq*word
        kernel.elementwise_input1.output_tensor_size = hidden*(seq_window+1)*word
        kernel.elementwise_input1.tiling = 4
    
    elif i == 7:
        kernel.name = "V_cache"
        kernel.id = 7
        kernel.fwd_bwd = 1
        kernel.type = 1
        kernel.config = -1
        kernel.elementwise_input1.memory_size = word*hidden*seq_window/tp
        kernel.elementwise_input1.outer = num_head
        kernel.elementwise_input1.M = head_dim
        kernel.elementwise_input1.N = seq_window+1
        kernel.elementwise_input1.input_tensor_size = hidden*seq*word
        kernel.elementwise_input1.output_tensor_size = hidden*(seq_window+1)*word
        kernel.elementwise_input1.tiling = 4

    elif i == 8:
        kernel.name = "MHA_GEMM_1"
        kernel.id = 8
        kernel.fwd_bwd = 1
        kernel.type = 1
        kernel.config = -1
        kernel.gemm_input1_input2.outer = num_head
        kernel.gemm_input1_input2.M = seq
        kernel.gemm_input1_input2.K = head_dim
        kernel.gemm_input1_input2.N = seq_window+1
        kernel.gemm_input1_input2.input_tensor_1_size = hidden*seq*word
        kernel.gemm_input1_input2.input_tensor_2_size = hidden*(seq_window+1)*word
        kernel.gemm_input1_input2.output_tensor_size = seq*(seq_window+1)*num_head*word
        kernel.gemm_input1_input2.tiling = 4
        
    elif i == 9:
        kernel.name = "SOFTMAX"
        kernel.id = 9
        kernel.fwd_bwd = 1
        kernel.type = 2
        kernel.config = -1
        kernel.elementwise_input1.outer = num_head
        kernel.elementwise_input1.M = seq
        kernel.elementwise_input1.N = seq_window+1
        kernel.elementwise_input1.input_tensor_size = seq*(seq_window+1)*num_head*word
        kernel.elementwise_input1.output_tensor_size = seq*(seq_window+1)*num_head*word
        kernel.elementwise_input1.tiling = 4

    elif i == 10:
        kernel.name = "DropOut_1"
        kernel.id = 10
        kernel.fwd_bwd = 1
        kernel.type = 2
        kernel.config = -1
        kernel.elementwise_input1.outer = num_head
        kernel.elementwise_input1.M = seq
        kernel.elementwise_input1.N = seq_window+1
        kernel.elementwise_input1.input_tensor_size = seq*(seq_window+1)*num_head*word
        kernel.elementwise_input1.output_tensor_size = seq*(seq_window+1)*num_head*word
        kernel.elementwise_input1.tiling = 4

    elif i == 11:
        kernel.name = "MHA_GEMM_2"
        kernel.id = 11
        kernel.fwd_bwd = 1
        kernel.type = 1
        kernel.config = -1
        kernel.gemm_input1_input2.outer = num_head
        kernel.gemm_input1_input2.M = head_dim
        kernel.gemm_input1_input2.K = seq_window+1
        kernel.gemm_input1_input2.N = seq
        kernel.gemm_input1_input2.input_tensor_1_size = hidden*(seq_window+1)*word
        kernel.gemm_input1_input2.input_tensor_2_size = seq*(seq_window+1)*num_head*word
        kernel.gemm_input1_input2.output_tensor_size = hidden*seq*word
        kernel.gemm_input1_input2.tiling = 4

    elif i == 12:
        kernel.name = "PROJ_GEMM"
        kernel.id = 12
        kernel.fwd_bwd = 1
        kernel.type = 1
        kernel.config = -1
        kernel.gemm_input1_weight.memory_size = hidden*hidden*word/tp
        kernel.gemm_input1_weight.outer = 1
        kernel.gemm_input1_weight.M = hidden
        kernel.gemm_input1_weight.K = hidden
        kernel.gemm_input1_weight.N = seq
        kernel.gemm_input1_weight.input_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.weight_tensor_size = hidden*hidden*word
        kernel.gemm_input1_weight.output_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.tiling = 4       

    elif i == 13:
        kernel.name = "DropOut_2"
        kernel.id = 13
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
        kernel.name = "Add_1"
        kernel.id = 14
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

    elif i == 15:
        kernel.name = "LayerNorm_2"
        kernel.id = 15
        kernel.fwd_bwd = 1
        kernel.type = 2
        kernel.config = -1  
        kernel.elementwise_input1.outer = 1
        kernel.elementwise_input1.M = hidden
        kernel.elementwise_input1.N = seq
        kernel.elementwise_input1.input_tensor_size = hidden*seq*word
        kernel.elementwise_input1.output_tensor_size = hidden*seq*word
        kernel.elementwise_input1.tiling = 4

    elif i == 16:
        kernel.name = "FFN0"
        kernel.id = 16
        kernel.fwd_bwd = 1
        kernel.type = 1
        kernel.config = -1
        kernel.gemm_input1_weight.memory_size = mlp_dim*hidden*word/tp
        kernel.gemm_input1_weight.outer = 1
        kernel.gemm_input1_weight.M = mlp_dim
        kernel.gemm_input1_weight.K = hidden
        kernel.gemm_input1_weight.N = seq
        kernel.gemm_input1_weight.input_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.weight_tensor_size = mlp_dim*hidden*word
        kernel.gemm_input1_weight.output_tensor_size = mlp_dim*seq*word
        kernel.gemm_input1_weight.tiling = 4
    
    elif i == 17:
        kernel.name = "GeLU"
        kernel.id = 17
        kernel.fwd_bwd = 1
        kernel.type = 2
        kernel.config = -1
        kernel.elementwise_input1.outer = 1
        kernel.elementwise_input1.M = mlp_dim
        kernel.elementwise_input1.N = seq 
        kernel.elementwise_input1.input_tensor_size = mlp_dim*seq*word
        kernel.elementwise_input1.output_tensor_size = mlp_dim*seq*word
        kernel.elementwise_input1.tiling = 4

    elif i == 18:
        kernel.name = "FFN1"
        kernel.id = 18
        kernel.fwd_bwd = 1
        kernel.type = 1
        kernel.config = -1
        kernel.gemm_input1_weight.memory_size = mlp_dim*hidden*word/tp
        kernel.gemm_input1_weight.outer = 1
        kernel.gemm_input1_weight.M = hidden
        kernel.gemm_input1_weight.K = mlp_dim
        kernel.gemm_input1_weight.N = seq
        kernel.gemm_input1_weight.input_tensor_size = mlp_dim*seq*word
        kernel.gemm_input1_weight.weight_tensor_size = mlp_dim*hidden*word
        kernel.gemm_input1_weight.output_tensor_size = hidden*seq*word
        kernel.gemm_input1_weight.tiling = 4

    elif i == 19:
        kernel.name = "DropOut_3"
        kernel.id = 19
        kernel.fwd_bwd = 1
        kernel.type = 2
        kernel.config = -1
        kernel.elementwise_input1.outer = 1
        kernel.elementwise_input1.M = hidden
        kernel.elementwise_input1.N = seq
        kernel.elementwise_input1.input_tensor_size = hidden*seq*word
        kernel.elementwise_input1.output_tensor_size = hidden*seq*word
        kernel.elementwise_input1.tiling = 4

    elif i == 20:
        kernel.name = "Add_2"
        kernel.id = 20
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
          

for i in range(1, 24):
    connection = dataflow_graph.connections.add()
    
    if i == 1:
        connection.id = i
        connection.startIdx = 1
        connection.endIdx = 2
    
    elif i == 2:
        connection.id = i
        connection.startIdx = 2
        connection.endIdx = 3
    
    elif i == 3:
        connection.id = i
        connection.startIdx = 2
        connection.endIdx = 4
    
    elif i == 4:
        connection.id = i
        connection.startIdx = 2
        connection.endIdx = 5

    elif i == 5:
        connection.id = i
        connection.startIdx = 4
        connection.endIdx = 6

    elif i == 6:
        connection.id = i
        connection.startIdx = 5
        connection.endIdx = 7

    elif i == 7:
        connection.id = i
        connection.startIdx = 3
        connection.endIdx = 8
    
    elif i == 8:
        connection.id = i
        connection.startIdx = 6
        connection.endIdx = 8
        connection.zero_out = True

    elif i == 9:
        connection.id = i
        connection.startIdx = 8
        connection.endIdx = 9

    elif i == 10:
        connection.id = i
        connection.startIdx = 9
        connection.endIdx = 10
    
    elif i == 11:
        connection.id = i
        connection.startIdx = 10
        connection.endIdx = 11
    
    elif i == 12:
        connection.id = i
        connection.startIdx = 7
        connection.endIdx = 11
        connection.zero_out = True
        
    elif i == 13:
        connection.id = i
        connection.startIdx = 11
        connection.endIdx = 12
    
    elif i == 14:
        connection.id = i
        connection.startIdx = 12
        connection.endIdx = 13
    
    elif i == 15:
        connection.id = i
        connection.startIdx = 13
        connection.endIdx = 14
    
    elif i == 16:
        connection.id = i
        connection.startIdx = 14
        connection.endIdx = 15
    
    elif i == 17:
        connection.id = i
        connection.startIdx = 1
        connection.endIdx = 14
    
    elif i == 18:
        connection.id = i
        connection.startIdx = 15
        connection.endIdx = 16
    
    elif i == 19:
        connection.id = i
        connection.startIdx = 16
        connection.endIdx = 17
    
    elif i == 20:
        connection.id = i
        connection.startIdx = 17
        connection.endIdx = 18
    
    elif i == 21:
        connection.id = i
        connection.startIdx = 18
        connection.endIdx = 19
        
    elif i == 22:
        connection.id = i
        connection.startIdx = 19
        connection.endIdx = 20
        
    elif i == 23:
        connection.id = i
        connection.startIdx = 14
        connection.endIdx = 20
        

# write to text file
with open('generator/LLM_fwd_decode.txt', "w") as file:
    text_format.PrintMessage(dataflow_graph, file)