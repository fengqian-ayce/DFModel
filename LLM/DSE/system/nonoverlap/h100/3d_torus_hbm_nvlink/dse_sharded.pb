
�&
;
Add_Prev_Layer ���������(0J�� �5  �LM  �L`�
:
LayerNorm_1 ���������(0J�� �5  �LM  �L`�
9
Q ���������(0:!���� �5  �LE @�NM  �L`�
9
K ���������(0:!���� �5  �LE @�NM  �L`�
9
V ���������(0:!���� �5  �LE @�NM  �L`�
A

MHA_GEMM_1 ���������(0B ��� �5  �LE  �LM  �N`�
6
SOFTMAX ���������(0J�� �5  �NM  �N`�
8
	DropOut_1 ���������(0J�� �5  �NM  �N`�
A

MHA_GEMM_2	 ���������(0B ��� �5  �LE  �NM  �L`�
H
	PROJ_GEMM
 ���������(0:(���� �5  �LE @�NM  �L`hu  �L�
8
	DropOut_2 ���������(0J�� �5  �LM  �L`�
:
Add_1	 ���������(0R�� �5  �LM  �L`��  �L
:
LayerNorm_2
 ���������(0J�� �5  �LM  �L`�
A
FFN0 ���������(0:&���� �5  �LE @�OM  �M`u   ��
3
GeLU ���������(0J�� �5  �MM  �M`�
C
FFN1 ���������(0:(���� �5  �ME @�OM  �L`hu  �L�
8
	DropOut_3 ���������(0J�� �5  �LM  �L`�
:
Add_2 ���������(0R�� �5  �LM  �L`��  �L
5
Loss_bwd ���������(0J�� �5  �LM  �L`�
<
DropOut_3_bwd ���������(0J�� �5  �LM  �L`�
@
FFN1_bwd ���������(0:!���� �5  �LE @�OM  �M`�
7
GeLU_bwd ���������(0J�� �5  �MM  �M`�
G
FFN0_bwd ���������(0:(���� �5  �ME @�OM  �L`hu  �L�
>
LayerNorm_2_bwd ���������(0J�� �5  �LM  �L`�
<
DropOut_2_bwd ���������(0J�� �5  �LM  �L`�
E
PROJ_GEMM_bwd ���������(0:!���� �5  �LE @�NM  �L`�
F
MHA_GEMM_2_bwd1 ���������(0B ��� �5  �LE  �LM  �N`�
F
MHA_GEMM_2_bwd2 ���������(0B ��� �5  �LE  �NM  �L`�
D
V_bwd	 ���������(0:(���� �5  �LE @�NM  �L`hu  �L�
<
DropOut_1_bwd	 ���������(0J�� �5  �NM  �N`�
:
SOFTMAX_bwd
 ���������(0J�� �5  �NM  �N`�
F
MHA_GEMM_1_bwd1  ���������(0B ��� �5  �NE  �LM  �L`�
F
MHA_GEMM_1_bwd2! ���������(0B ��� �5  �NE  �LM  �L`�
=
Q_bwd" ���������(0:!���� �5  �LE @�NM  �L`�
=
K_bwd# ���������(0:!���� �5  �LE @�NM  �L`�
U
FFN1_bwd_weight_update$ ���������(0B(��� ��5  �LE  �MM @�O`hu @�O�
U
FFN0_bwd_weight_update% ���������(0B(��� ��5  �ME  �LM @�O`hu @�O�
Z
PROJ_GEMM_bwd_weight_update& ���������(0B(��� ��5  �LE  �LM @�N`hu @�N�
R
V_bwd_weight_update'	 ���������(0B(��� ��5  �LE  �LM @�N`hu @�N�
R
K_bwd_weight_update( ���������(0B(��� ��5  �LE  �LM @�N`hu @�N�
R
Q_bwd_weight_update) ���������(0B(��� ��5  �LE  �LM @�N`hu @�N�/-   �0=  �LPZAdd_Prev_LayerbLayerNorm_10=  �LPZLayerNorm_1bQ0=  �LPZLayerNorm_1bK0=  �LPZLayerNorm_1bV0=  �LPZQb
MHA_GEMM_10=  �LPZKb
MHA_GEMM_1"0=  �NPZ
MHA_GEMM_1bSOFTMAX!0=  �NPZSOFTMAXb	DropOut_1	0=  �LP	ZVb
MHA_GEMM_2$	0=  �NP
Z	DropOut_1b
MHA_GEMM_2$	
0=  �LPZ
MHA_GEMM_2b	PROJ_GEMM(
-   �0=  �LPZ	PROJ_GEMMb	DropOut_2$-   �0=  �LPZ	DropOut_2bAdd_1)-   �0
=  �LPZAdd_Prev_LayerbAdd_1&-   �0=  �LPZAdd_1bLayerNorm_2 0=  �LPZLayerNorm_2bFFN0-   �0=  �MPZFFN0bGeLU0=  �MPZGeLUbFFN1#-   �0=  �LPZFFN1b	DropOut_30=  �LPZ	DropOut_3bAdd_20=  �LPZAdd_1bAdd_2&0=  �LPZLoss_bwdbDropOut_3_bwd&0=  �LPZDropOut_3_bwdbFFN1_bwd&-   �0=  �MPZFFN1_bwdbGeLU_bwd!0=  �MPZGeLU_bwdbFFN0_bwd--   �0=  �LPZFFN0_bwdbLayerNorm_2_bwd2-   �0=  �LPZLayerNorm_2_bwdbDropOut_2_bwd+0=  �LPZDropOut_2_bwdbPROJ_GEMM_bwd-0=  �LPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd1!0=  �LPZVbMHA_GEMM_2_bwd1-0=  �LPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd2)0=  �NP Z	DropOut_1bMHA_GEMM_2_bwd2-0=  �NP!ZMHA_GEMM_2_bwd1bDropOut_1_bwd%0=  �LP"ZMHA_GEMM_2_bwd2bV_bwd)0=  �NP#ZDropOut_1_bwdbSOFTMAX_bwd+ 0=  �NP$ZSOFTMAX_bwdbMHA_GEMM_1_bwd1+!0=  �NP%ZSOFTMAX_bwdbMHA_GEMM_1_bwd2! 0
=  �LP&ZKbMHA_GEMM_1_bwd1!!0
=  �LP'ZQbMHA_GEMM_1_bwd2% "0=  �LP(ZMHA_GEMM_1_bwd1bQ_bwd%!#0=  �LP)ZMHA_GEMM_1_bwd2bK_bwd4$0=  �LP*ZDropOut_3_bwdbFFN1_bwd_weight_update+$0=  �MP+ZGeLUbFFN1_bwd_weight_update/%0	=  �MP,ZGeLU_bwdbFFN0_bwd_weight_update2%0=  �LP-ZLayerNorm_2bFFN0_bwd_weight_update9&0=  �LP.ZDropOut_2_bwdbPROJ_GEMM_bwd_weight_update6	&0=  �LP/Z
MHA_GEMM_2bPROJ_GEMM_bwd_weight_update3'0=  �LP0ZMHA_GEMM_2_bwd2bV_bwd_weight_update4'-   �0	=  �LP1ZLayerNorm_1bV_bwd_weight_update3!(0=  �LP2ZMHA_GEMM_1_bwd2bK_bwd_weight_update/(0=  �LP3ZLayerNorm_1bK_bwd_weight_update3 )0=  �LP4ZMHA_GEMM_1_bwd1bQ_bwd_weight_update/)0=  �LP5ZLayerNorm_1bQ_bwd_weight_updateO�� -  �L5�"�?b*��@��  �C�  HB�  HB�TP�PP�DP�
  @E  �Q2   @  �A  �?% � G-�d*<5��T==��T=E��T=M�[f=U �;D"%:���� �(�X�`h�p�fff?�*
o�:�