import os
import multiprocessing


os.system('./run.sh SSM/Hyena/sweep_SRAM/2TB_dram/regular_fft/600MB_sram_3TB_dram/ > SSM/Hyena/sweep_SRAM/2TB_dram/regular_fft/600MB_sram_3TB_dram/log.txt')
os.system('./run.sh SSM/Hyena/sweep_SRAM/2TB_dram/regular_fft/1200MB_sram_3TB_dram/ > SSM/Hyena/sweep_SRAM/2TB_dram/regular_fft/1200MB_sram_3TB_dram/log.txt')

os.system('./run.sh SSM/Hyena/sweep_SRAM/2TB_dram/gemm_fft/600MB_sram_3TB_dram/ > SSM/Hyena/sweep_SRAM/2TB_dram/gemm_fft/600MB_sram_3TB_dram/log.txt')
os.system('./run.sh SSM/Hyena/sweep_SRAM/2TB_dram/gemm_fft/1200MB_sram_3TB_dram/ > SSM/Hyena/sweep_SRAM/2TB_dram/gemm_fft/1200MB_sram_3TB_dram/log.txt')

os.system('./run.sh SSM/Hyena/sweep_SRAM/2TB_dram/vector_fft/600MB_sram_3TB_dram/ > SSM/Hyena/sweep_SRAM/2TB_dram/vector_fft/600MB_sram_3TB_dram/log.txt')
os.system('./run.sh SSM/Hyena/sweep_SRAM/2TB_dram/vector_fft/1200MB_sram_3TB_dram/ > SSM/Hyena/sweep_SRAM/2TB_dram/vector_fft/1200MB_sram_3TB_dram/log.txt')

