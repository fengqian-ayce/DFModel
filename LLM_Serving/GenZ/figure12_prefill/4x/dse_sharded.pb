
ţ
;
Add_Prev_Layer ˙˙˙˙˙˙˙˙˙(0J`  5 `jMM `jM` 
:
LayerNorm_1 ˙˙˙˙˙˙˙˙˙(0J`  5 `jMM `jM` 
8
Q ˙˙˙˙˙˙˙˙˙(0: ``  5 `jME  MM `jM` 
8
K ˙˙˙˙˙˙˙˙˙(0: ``  5 `jME  MM `jM` 
8
V ˙˙˙˙˙˙˙˙˙(0: ``  5 `jME  MM `jM` 
B

MHA_GEMM_1 ˙˙˙˙˙˙˙˙˙(0B!`   5 `jME `jMMQ` 
7
SOFTMAX ˙˙˙˙˙˙˙˙˙(0J`   5QMQ` 
9
	DropOut_1 ˙˙˙˙˙˙˙˙˙(0J`   5QMQ` 
B

MHA_GEMM_2	 ˙˙˙˙˙˙˙˙˙(0B!`   5 `jMEQM `jM` 
G
	PROJ_GEMM
 ˙˙˙˙˙˙˙˙˙(0:'``  5 `jME  MM `jM`hu `jM 
8
	DropOut_2 ˙˙˙˙˙˙˙˙˙(0J`  5 `jMM `jM` 
:
Add_1	 ˙˙˙˙˙˙˙˙˙(0R`  5 `jMM `jM` ő `jM
:
LayerNorm_2
 ˙˙˙˙˙˙˙˙˙(0J`  5 `jMM `jM` 
<
FFN0 ˙˙˙˙˙˙˙˙˙(0:!`  5 `jME  NM `jN` 
4
GeLU ˙˙˙˙˙˙˙˙˙(0J  5 `jNM `jN` 
C
FFN1 ˙˙˙˙˙˙˙˙˙(0:(`  5 `jNE  NM `jM`hu `jM 
8
	DropOut_3 ˙˙˙˙˙˙˙˙˙(0J`  5 `jMM `jM` 
:
Add_2 ˙˙˙˙˙˙˙˙˙(0R`  5 `jMM `jM` ő `jM2-   0= `jMPZAdd_Prev_LayerbLayerNorm_1  0= `jMPZLayerNorm_1bQ  0= `jMPZLayerNorm_1bK  0= `jMPZLayerNorm_1bV 0= `jMPZQb
MHA_GEMM_1 0= `jMPZKb
MHA_GEMM_1 %0=QPZ
MHA_GEMM_1bSOFTMAX $0=QPZSOFTMAXb	DropOut_1 	0= `jMP	ZVb
MHA_GEMM_2 '	0=QP
Z	DropOut_1b
MHA_GEMM_2 '	
0= `jMPZ
MHA_GEMM_2b	PROJ_GEMM +
-   0= `jMPZ	PROJ_GEMMb	DropOut_2 '-   0= `jMPZ	DropOut_2bAdd_1 ,-   0
= `jMPZAdd_Prev_LayerbAdd_1 )-   0= `jMPZAdd_1bLayerNorm_2 (-   0= `jMPZLayerNorm_2bFFN0 !-   0= `jNPZFFN0bGeLU !-   0= `jNPZGeLUbFFN1 &-   0= `jMPZFFN1b	DropOut_3 '-   0= `jMPZ	DropOut_3bAdd_2 #-   0= `jMPZAdd_1bAdd_2 Fľ -  ČQU  ?˝E  úD: ¨ľ  D˝  DÂTPĘPP˘
  zE   Q(   @  ŔA  ?% č G-Ăd*<EôýT=M(a&>U ;D"-:``  (`X`h xpx  ? *o:´