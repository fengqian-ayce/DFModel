
�*
9
Add_Prev_Layer(0J!�� �5  �LM  �L`������
9
LayerNorm_1 (0J �� �5  �LM  �L`�����
=
Q (0:.���� �5  �LE  HNM  �L`��������
=
K (0:.���� �5  �LE  HNM  �L`��������
=
V (0:.���� �5  �LE  HNM  �L`��������
E

MHA_GEMM_1 (0B-��� �5  �LE  �LM  �N`��������
6
SOFTMAX (0J!�� �5  �NM  �N`������
8
	DropOut_1 (0J!�� �5  �NM  �N`������
D

MHA_GEMM_2	 (0B,��� �5  �LE  �NM  �L`�������
L
	PROJ_GEMM
 	(0:5���� �5  �LE  HNM  �L`hu  �L��������
8
	DropOut_2 
(0J!�� �5  �LM  �L`������
:
Add_1	 (0R'�� �5  �LM  �L`�������  �L
:
LayerNorm_2
 (0J!�� �5  �LM  �L`������
E
FFN0 (0:3���� �5  �LE  HOM  �M`u   ���P������
2
GeLU (0J �� �5  �MM  �M`��P���
G
FFN1 (0:5���� �5  �ME  HOM  �L`hu  �L�����P���
7
	DropOut_3 (0J �� �5  �LM  �L`�����
9
Add_2 (0R&�� �5  �LM  �L`������  �L
5
Loss_bwd (0J!�� �5  �LM  �L`������
<
DropOut_3_bwd (0J!�� �5  �LM  �L`������
D
FFN1_bwd (0:.���� �5  �LE  HOM  �M`��P������
6
GeLU_bwd (0J �� �5  �MM  �M`��P���
K
FFN0_bwd (0:5���� �5  �ME  HOM  �L`hu  �L�����P���
>
LayerNorm_2_bwd (0J!�� �5  �LM  �L`������
<
DropOut_2_bwd (0J!�� �5  �LM  �L`������
I
PROJ_GEMM_bwd (0:.���� �5  �LE  HNM  �L`��������
J
MHA_GEMM_2_bwd1 (0B-��� �5  �LE  �LM  �N`��������
I
MHA_GEMM_2_bwd2 (0B,��� �5  �LE  �NM  �L`�������
H
V_bwd	 (0:5���� �5  �LE  HNM  �L`hu  �L��������
<
DropOut_1_bwd	 (0J!�� �5  �NM  �N`������
:
SOFTMAX_bwd
 (0J!�� �5  �NM  �N`������
I
MHA_GEMM_1_bwd1  (0B,��� �5  �NE  �LM  �L`�������
I
MHA_GEMM_1_bwd2!  (0B,��� �5  �NE  �LM  �L`�������
A
Q_bwd" !(0:.���� �5  �LE  HNM  �L`��������
A
K_bwd# "(0:.���� �5  �LE  HNM  �L`��������
Y
FFN1_bwd_weight_update$ #(0B5��� ��5  �LE  �MM  HO`hu  HO��P������
Y
FFN0_bwd_weight_update% $(0B5��� ��5  �ME  �LM  HO`hu  HO��������
^
PROJ_GEMM_bwd_weight_update& %(0B5��� ��5  �LE  �LM  HN`hu  HN��������
V
V_bwd_weight_update'	 &(0B5��� ��5  �LE  �LM  HN`hu  HN��������
V
K_bwd_weight_update( '(0B5��� ��5  �LE  �LM  HN`hu  HN��������
V
Q_bwd_weight_update) ((0B5��� ��5  �LE  �LM  HN`hu  HN��������20=  �LE  �JPZAdd_Prev_LayerbLayerNorm_1�%0=  �LE   IPZLayerNorm_1bQ�%0=  �LE   IPZLayerNorm_1bK�%0=  �LE   IPZLayerNorm_1bV�$0=  �LE   IPZQb
MHA_GEMM_1�$0=  �LE   IPZKb
MHA_GEMM_1�*0=  �NE   KPZ
MHA_GEMM_1bSOFTMAX�)0=  �NE   KPZSOFTMAXb	DropOut_1�$	0=  �LE   IP	ZVb
MHA_GEMM_2�,	0=  �NE   KP
Z	DropOut_1b
MHA_GEMM_2�,	
0=  �LE   IPZ
MHA_GEMM_2b	PROJ_GEMM�+
0=  �LE  �JPZ	PROJ_GEMMb	DropOut_2�'0=  �LE  �JPZ	DropOut_2bAdd_1�,0
=  �LE  �JPZAdd_Prev_LayerbAdd_1�)0=  �LE  �JPZAdd_1bLayerNorm_2�(0=  �LE  �JPZLayerNorm_2bFFN0�!0=  �ME   JPZFFN0bGeLU�!0=  �ME   JPZGeLUbFFN1�&0=  �LE �JPZFFN1b	DropOut_3�'0=  �LE   IPZ	DropOut_3bAdd_2�#0=  �LE  �JPZAdd_1bAdd_2�.0=  �LE  �JPZLoss_bwdbDropOut_3_bwd�.0=  �LE  �JPZDropOut_3_bwdbFFN1_bwd�)0=  �ME   JPZFFN1_bwdbGeLU_bwd�)0=  �ME   JPZGeLU_bwdbFFN0_bwd�00=  �LE  �JPZFFN0_bwdbLayerNorm_2_bwd�50=  �LE  �JPZLayerNorm_2_bwdbDropOut_2_bwd�30=  �LE  �JPZDropOut_2_bwdbPROJ_GEMM_bwd�50=  �LE   IPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd1�)0=  �LE   IPZVbMHA_GEMM_2_bwd1�50=  �LE   IPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd2�10=  �NE   KP Z	DropOut_1bMHA_GEMM_2_bwd2�50=  �NE   KP!ZMHA_GEMM_2_bwd1bDropOut_1_bwd�-0=  �LE   IP"ZMHA_GEMM_2_bwd2bV_bwd�10=  �NE   KP#ZDropOut_1_bwdbSOFTMAX_bwd�3 0=  �NE   KP$ZSOFTMAX_bwdbMHA_GEMM_1_bwd1�3!0=  �NE   KP%ZSOFTMAX_bwdbMHA_GEMM_1_bwd2�) 0
=  �LE   IP&ZKbMHA_GEMM_1_bwd1�)!0
=  �LE   IP'ZQbMHA_GEMM_1_bwd2�- "0=  �LE   IP(ZMHA_GEMM_1_bwd1bQ_bwd�-!#0=  �LE @!IP)ZMHA_GEMM_1_bwd2bK_bwd�<$0=  �LE  �JP*ZDropOut_3_bwdbFFN1_bwd_weight_update�3$0=  �ME   JP+ZGeLUbFFN1_bwd_weight_update�7%0	=  �ME   JP,ZGeLU_bwdbFFN0_bwd_weight_update�:%0=  �LE  �JP-ZLayerNorm_2bFFN0_bwd_weight_update�A&0=  �LE  �JP.ZDropOut_2_bwdbPROJ_GEMM_bwd_weight_update�>	&0=  �LE   IP/Z
MHA_GEMM_2bPROJ_GEMM_bwd_weight_update�;'0=  �LE   IP0ZMHA_GEMM_2_bwd2bV_bwd_weight_update�7'0	=  �LE   IP1ZLayerNorm_1bV_bwd_weight_update�;!(0=  �LE @!IP2ZMHA_GEMM_1_bwd2bK_bwd_weight_update�7(0=  �LE   IP3ZLayerNorm_1bK_bwd_weight_update�; )0=  �LE   IP4ZMHA_GEMM_1_bwd1bQ_bwd_weight_update�7)0=  �LE   IP5ZLayerNorm_1bQ_bwd_weight_update�O�� -  �LU�z�?b*��#�	�  �C�  HA�  HA�TP�PP�DP�
 ��D  �Q-   @  �A  �?% � G-�d*<5��T=E��T=M(a&>U �;D"#:���� �(iX�`xp���*o�:�