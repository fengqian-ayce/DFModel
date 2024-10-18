import os
import multiprocessing


os.system('./run.sh SSM/Hyena/sweep_SRAM/8TB_dram/regular_fft/600MB_sram/ > SSM/Hyena/sweep_SRAM/8TB_dram/regular_fft/600MB_sram/log.txt')
os.system('./run.sh SSM/Hyena/sweep_SRAM/8TB_dram/regular_fft/1200MB_sram/ > SSM/Hyena/sweep_SRAM/8TB_dram/regular_fft/1200MB_sram/log.txt')

os.system('./run.sh SSM/Hyena/sweep_SRAM/8TB_dram/gemm_fft/600MB_sram/ > SSM/Hyena/sweep_SRAM/8TB_dram/gemm_fft/600MB_sram/log.txt')
os.system('./run.sh SSM/Hyena/sweep_SRAM/8TB_dram/gemm_fft/1200MB_sram/ > SSM/Hyena/sweep_SRAM/8TB_dram/gemm_fft/1200MB_sram/log.txt')

os.system('./run.sh SSM/Hyena/sweep_SRAM/8TB_dram/vector_fft/600MB_sram/ > SSM/Hyena/sweep_SRAM/8TB_dram/vector_fft/600MB_sram/log.txt')
os.system('./run.sh SSM/Hyena/sweep_SRAM/8TB_dram/vector_fft/1200MB_sram/ > SSM/Hyena/sweep_SRAM/8TB_dram/vector_fft/1200MB_sram/log.txt')

