
�&
:
Add_Prev_Layer ���������(0J�( � 5   LM   L`�
9
LayerNorm_1 ���������(0J�( � 5   LM   L`�
7
Q ���������(0:(��( � 5   LE  HLM   L`�
7
K ���������(0:(��( � 5   LE  HLM   L`�
7
V ���������(0:(��( � 5   LE  HLM   L`�
@

MHA_GEMM_1 ���������(0B(� � � 5   LE   LM  �N`�
5
SOFTMAX ���������(0J(�  � 5  �NM  �N`�
7
	DropOut_1 ���������(0J(�  � 5  �NM  �N`�
@

MHA_GEMM_2	 ���������(0B(��  � 5   LE  �NM   L`�
F
	PROJ_GEMM
 ���������(0:&�(�( � 5   LE  HLM   L`hu   L�
7
	DropOut_2 ���������(0J�( � 5   LM   L`�
9
Add_1	 ���������(0R�( � 5   LM   L`��   L
9
LayerNorm_2
 ���������(0J�( � 5   LM   L`�
@
FFN0 ���������(0:%���( � 5   LE  HMM   M`u   ��
3
GeLU ���������(0J�� � 5   MM   M`�
B
FFN1 ���������(0:'�(�� � 5   ME  HMM   L`hu   L�
7
	DropOut_3 ���������(0J�( � 5   LM   L`�
9
Add_2 ���������(0R�( � 5   LM   L`��   L
4
Loss_bwd ���������(0J�( � 5   LM   L`�
;
DropOut_3_bwd ���������(0J�( � 5   LM   L`�
?
FFN1_bwd ���������(0: ���( � 5   LE  HMM   M`�
7
GeLU_bwd ���������(0J�� � 5   MM   M`�
F
FFN0_bwd ���������(0:'�(�� � 5   ME  HMM   L`hu   L�
=
LayerNorm_2_bwd ���������(0J�( � 5   LM   L`�
;
DropOut_2_bwd ���������(0J�( � 5   LM   L`�
C
PROJ_GEMM_bwd ���������(0:�(�( � 5   LE  HLM   L`�
E
MHA_GEMM_2_bwd1 ���������(0B(� � � 5   LE   LM  �N`�
E
MHA_GEMM_2_bwd2 ���������(0B(��  � 5   LE  �NM   L`�
B
V_bwd	 ���������(0:&�(�( � 5   LE  HLM   L`hu   L�
;
DropOut_1_bwd	 ���������(0J(�  � 5  �NM  �N`�
9
SOFTMAX_bwd
 ���������(0J(�  � 5  �NM  �N`�
E
MHA_GEMM_1_bwd1  ���������(0B(��  � 5  �NE   LM   L`�
E
MHA_GEMM_1_bwd2! ���������(0B(��  � 5  �NE   LM   L`�
;
Q_bwd" ���������(0:(��( � 5   LE  HLM   L`�
;
K_bwd# ���������(0:(��( � 5   LE  HLM   L`�
T
FFN1_bwd_weight_update$ ���������(0B'���  �(5   LE   MM  HM`hu  HM�
T
FFN0_bwd_weight_update% ���������(0B'�(�  ��5   ME   LM  HM`hu  HM�
X
PROJ_GEMM_bwd_weight_update& ���������(0B&�(�  �(5   LE   LM  HL`hu  HL�
P
V_bwd_weight_update'	 ���������(0B&�(�  �(5   LE   LM  HL`hu  HL�
P
K_bwd_weight_update( ���������(0B&�(�  �(5   LE   LM  HL`hu  HL�
P
Q_bwd_weight_update) ���������(0B&�(�  �(5   LE   LM  HL`hu  HL�-0=   LPZAdd_Prev_LayerbLayerNorm_1� 0=   LPZLayerNorm_1bQ� 0=   LPZLayerNorm_1bK� 0=   LPZLayerNorm_1bV�0=   LPZQb
MHA_GEMM_1�0=   LPZKb
MHA_GEMM_1�%0=  �NPZ
MHA_GEMM_1bSOFTMAX�$0=  �NPZSOFTMAXb	DropOut_1�	0=   LP	ZVb
MHA_GEMM_2�'	0=  �NP
Z	DropOut_1b
MHA_GEMM_2�'	
0=   LPZ
MHA_GEMM_2b	PROJ_GEMM�&
0=   LPZ	PROJ_GEMMb	DropOut_2�"0=   LPZ	DropOut_2bAdd_1�'0
=   LPZAdd_Prev_LayerbAdd_1�$0=   LPZAdd_1bLayerNorm_2�#0=   LPZLayerNorm_2bFFN0�0=   MPZFFN0bGeLU�0=   MPZGeLUbFFN1�!0=   LPZFFN1b	DropOut_3�"0=   LPZ	DropOut_3bAdd_2�0=   LPZAdd_1bAdd_2�)0=   LPZLoss_bwdbDropOut_3_bwd�)0=   LPZDropOut_3_bwdbFFN1_bwd�$0=   MPZFFN1_bwdbGeLU_bwd�$0=   MPZGeLU_bwdbFFN0_bwd�+0=   LPZFFN0_bwdbLayerNorm_2_bwd�00=   LPZLayerNorm_2_bwdbDropOut_2_bwd�.0=   LPZDropOut_2_bwdbPROJ_GEMM_bwd�00=   LPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd1�$0=   LPZVbMHA_GEMM_2_bwd1�00=   LPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd2�,0=  �NP Z	DropOut_1bMHA_GEMM_2_bwd2�00=  �NP!ZMHA_GEMM_2_bwd1bDropOut_1_bwd�(0=   LP"ZMHA_GEMM_2_bwd2bV_bwd�,0=  �NP#ZDropOut_1_bwdbSOFTMAX_bwd�. 0=  �NP$ZSOFTMAX_bwdbMHA_GEMM_1_bwd1�.!0=  �NP%ZSOFTMAX_bwdbMHA_GEMM_1_bwd2�$ 0
=   LP&ZKbMHA_GEMM_1_bwd1�$!0
=   LP'ZQbMHA_GEMM_1_bwd2�( "0=   LP(ZMHA_GEMM_1_bwd1bQ_bwd�(!#0=   LP)ZMHA_GEMM_1_bwd2bK_bwd�7$0=   LP*ZDropOut_3_bwdbFFN1_bwd_weight_update�.$0=   MP+ZGeLUbFFN1_bwd_weight_update�2%0	=   MP,ZGeLU_bwdbFFN0_bwd_weight_update�5%0=   LP-ZLayerNorm_2bFFN0_bwd_weight_update�<&0=   LP.ZDropOut_2_bwdbPROJ_GEMM_bwd_weight_update�9	&0=   LP/Z
MHA_GEMM_2bPROJ_GEMM_bwd_weight_update�6'0=   LP0ZMHA_GEMM_2_bwd2bV_bwd_weight_update�2'0	=   LP1ZLayerNorm_1bV_bwd_weight_update�6!(0=   LP2ZMHA_GEMM_1_bwd2bK_bwd_weight_update�2(0=   LP3ZLayerNorm_1bK_bwd_weight_update�6 )0=   LP4ZMHA_GEMM_1_bwd1bQ_bwd_weight_update�2)0=   LP5ZLayerNorm_1bQ_bwd_weight_update�-�  -  �MU  �?"��  HB�DP�  �B-   @  �A  �?% � G-�d*<5��T=E��T=M(a&>U �;D" :�(�( � ((X�@`xpx��*o�:�