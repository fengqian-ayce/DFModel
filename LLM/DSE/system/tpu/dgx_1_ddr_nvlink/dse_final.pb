
�)
9
Add_Prev_Layer(0J!�� �5  �LM  �L`������
:
LayerNorm_1 (0J!�� �5  �LM  �L`������
=
Q (0:.���� �5  �LE @�NM  �L`��������
=
K (0:.���� �5  �LE @�NM  �L`��������
=
V (0:.���� �5  �LE @�NM  �L`��������
E

MHA_GEMM_1 (0B-��� �5  �LE  �LM  �N`��������
6
SOFTMAX (0J!�� �5  �NM  �N`������
8
	DropOut_1 (0J!�� �5  �NM  �N`������
D

MHA_GEMM_2	 (0B,��� �5  �LE  �NM  �L`�������
L
	PROJ_GEMM
 	(0:5���� �5  �LE @�NM  �L`hu  �L��������
8
	DropOut_2 
(0J!�� �5  �LM  �L`������
:
Add_1	 (0R'�� �5  �LM  �L`�������  �L
:
LayerNorm_2
 (0J!�� �5  �LM  �L`������
E
FFN0 (0:3���� �5  �LE @�OM  �M`u   ���d������
2
GeLU (0J �� �5  �MM  �M`��d���
G
FFN1 (0:5���� �5  �ME @�OM  �L`hu  �L�����d���
8
	DropOut_3 (0J!�� �5  �LM  �L`������
9
Add_2 (0R&�� �5  �LM  �L`������  �L
5
Loss_bwd (0J!�� �5  �LM  �L`������
<
DropOut_3_bwd (0J!�� �5  �LM  �L`������
D
FFN1_bwd (0:.���� �5  �LE @�OM  �M`��d������
6
GeLU_bwd (0J �� �5  �MM  �M`��d���
K
FFN0_bwd (0:5���� �5  �ME @�OM  �L`hu  �L�����d���
>
LayerNorm_2_bwd (0J!�� �5  �LM  �L`������
<
DropOut_2_bwd (0J!�� �5  �LM  �L`������
I
PROJ_GEMM_bwd (0:.���� �5  �LE @�NM  �L`��������
J
MHA_GEMM_2_bwd1 (0B-��� �5  �LE  �LM  �N`��������
I
MHA_GEMM_2_bwd2 (0B,��� �5  �LE  �NM  �L`�������
H
V_bwd	 (0:5���� �5  �LE @�NM  �L`hu  �L��������
<
DropOut_1_bwd	 (0J!�� �5  �NM  �N`������
:
SOFTMAX_bwd
 (0J!�� �5  �NM  �N`������
I
MHA_GEMM_1_bwd1  (0B,��� �5  �NE  �LM  �L`����e���
I
MHA_GEMM_1_bwd2!  (0B,��� �5  �NE  �LM  �L`�������
A
Q_bwd" !(0:.���� �5  �LE @�NM  �L`��������
A
K_bwd# "(0:.���� �5  �LE @�NM  �L`��������
Y
FFN1_bwd_weight_update$ #(0B5��� ��5  �LE  �MM @�O`hu @�O��d������
Y
FFN0_bwd_weight_update% $(0B5��� ��5  �ME  �LM @�O`hu @�O��������
^
PROJ_GEMM_bwd_weight_update& %(0B5��� ��5  �LE  �LM @�N`hu @�N��������
V
V_bwd_weight_update'	 &(0B5��� ��5  �LE  �LM @�N`hu @�N��������
V
K_bwd_weight_update( '(0B5��� ��5  �LE  �LM @�N`hu @�N��������
V
Q_bwd_weight_update) ((0B5��� ��5  �LE  �LM @�N`hu @�N��������4-   �0=  �LE�)�KPZAdd_Prev_LayerbLayerNorm_1"0=  �LE (�KPZLayerNorm_1bQ"0=  �LE (�KPZLayerNorm_1bK"0=  �LE (�KPZLayerNorm_1bV!0=  �LE ( JPZQb
MHA_GEMM_1!0=  �LE ( JPZKb
MHA_GEMM_1'0=  �NE   LPZ
MHA_GEMM_1bSOFTMAX&0=  �NE   LPZSOFTMAXb	DropOut_1!	0=  �LE�4 JP	ZVb
MHA_GEMM_2)	0=  �NE   LP
Z	DropOut_1b
MHA_GEMM_2)	
0=  �LE ( JPZ
MHA_GEMM_2b	PROJ_GEMM-
-   �0=  �LE (�KPZ	PROJ_GEMMb	DropOut_2)-   �0=  �LE�)�KPZ	DropOut_2bAdd_1.-   �0
=  �LE�)�KPZAdd_Prev_LayerbAdd_1+-   �0=  �LE�)�KPZAdd_1bLayerNorm_2%0=  �LE�)�KPZLayerNorm_2bFFN0#-   �0=  �ME ( KPZFFN0bGeLU0=  �ME � KPZGeLUbFFN1(-   �0=  �LE (�KPZFFN1b	DropOut_3$0=  �LE�)�KPZ	DropOut_3bAdd_2 0=  �LE�)�KPZAdd_1bAdd_2+0=  �LE ��KPZLoss_bwdbDropOut_3_bwd+0=  �LE ��KPZDropOut_3_bwdbFFN1_bwd+-   �0=  �ME ( KPZFFN1_bwdbGeLU_bwd&0=  �ME � KPZGeLU_bwdbFFN0_bwd2-   �0=  �LE (�KPZFFN0_bwdbLayerNorm_2_bwd7-   �0=  �LE ��KPZLayerNorm_2_bwdbDropOut_2_bwd00=  �LE�)�KPZDropOut_2_bwdbPROJ_GEMM_bwd20=  �LE ( JPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd1&0=  �LE�4 JPZVbMHA_GEMM_2_bwd120=  �LE ( JPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd2.0=  �NE   LP Z	DropOut_1bMHA_GEMM_2_bwd220=  �NE   LP!ZMHA_GEMM_2_bwd1bDropOut_1_bwd*0=  �LE ( JP"ZMHA_GEMM_2_bwd2bV_bwd.0=  �NE   LP#ZDropOut_1_bwdbSOFTMAX_bwd0 0=  �NE   LP$ZSOFTMAX_bwdbMHA_GEMM_1_bwd10!0=  �NE   LP%ZSOFTMAX_bwdbMHA_GEMM_1_bwd2& 0
=  �LE ( JP&ZKbMHA_GEMM_1_bwd1&!0
=  �LE ( JP'ZQbMHA_GEMM_1_bwd2* "0=  �LE ( JP(ZMHA_GEMM_1_bwd1bQ_bwd*!#0=  �LE ( JP)ZMHA_GEMM_1_bwd2bK_bwd9$0=  �LE ��KP*ZDropOut_3_bwdbFFN1_bwd_weight_update0$0=  �ME � KP+ZGeLUbFFN1_bwd_weight_update4%0	=  �ME � KP,ZGeLU_bwdbFFN0_bwd_weight_update7%0=  �LE�)�KP-ZLayerNorm_2bFFN0_bwd_weight_update>&0=  �LE�)�KP.ZDropOut_2_bwdbPROJ_GEMM_bwd_weight_update;	&0=  �LE ( JP/Z
MHA_GEMM_2bPROJ_GEMM_bwd_weight_update8'0=  �LE ( JP0ZMHA_GEMM_2_bwd2bV_bwd_weight_update9'-   �0	=  �LE (�KP1ZLayerNorm_1bV_bwd_weight_update8!(0=  �LE ( JP2ZMHA_GEMM_1_bwd2bK_bwd_weight_update4(0=  �LE (�KP3ZLayerNorm_1bK_bwd_weight_update8 )0=  �LE ( JP4ZMHA_GEMM_1_bwd1bQ_bwd_weight_update4)0=  �LE (�KP5ZLayerNorm_1bQ_bwd_weight_updateC�� �-   M5ff�?J����  �C�  HB�TP�PP�
  �C  �S-   @  �A  �?%�֋E-�d*<5��T=E��T=M(a&>U��*C"(:���� �(�X�`h�p��fff?�*
o�:�