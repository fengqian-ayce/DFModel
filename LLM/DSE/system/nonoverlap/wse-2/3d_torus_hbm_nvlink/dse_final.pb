
�(
9
Add_Prev_Layer(0J!�� �5  �LM  �L`������
8
LayerNorm_1(0J!�� �5  �LM  �L`������
;
Q(0:.���� �5  �LE @�NM  �L`��������
;
K(0:.���� �5  �LE @�NM  �L`��������
;
V(0:.���� �5  �LE @�NM  �L`��������
C

MHA_GEMM_1(0B-��� �5  �LE  �LM  �N`��������
4
SOFTMAX(0J!�� �5  �NM  �N`������
6
	DropOut_1(0J!�� �5  �NM  �N`������
B

MHA_GEMM_2	(0B,��� �5  �LE  �NM  �L`�������
J
	PROJ_GEMM
(0:5���� �5  �LE @�NM  �L`hu  �L��������
6
	DropOut_2(0J!�� �5  �LM  �L`������
8
Add_1	(0R'�� �5  �LM  �L`�������  �L
8
LayerNorm_2
(0J!�� �5  �LM  �L`������
C
FFN0(0:3���� �5  �LE @�OM  �M`u   ���d������
0
GeLU(0J �� �5  �MM  �M`��d���
E
FFN1(0:5���� �5  �ME @�OM  �L`hu  �L�����d���
6
	DropOut_3(0J!�� �5  �LM  �L`������
7
Add_2(0R&�� �5  �LM  �L`������  �L
3
Loss_bwd(0J!�� �5  �LM  �L`������
:
DropOut_3_bwd(0J!�� �5  �LM  �L`������
B
FFN1_bwd(0:.���� �5  �LE @�OM  �M`��d������
4
GeLU_bwd(0J �� �5  �MM  �M`��d���
I
FFN0_bwd(0:5���� �5  �ME @�OM  �L`hu  �L�����d���
<
LayerNorm_2_bwd(0J!�� �5  �LM  �L`������
:
DropOut_2_bwd(0J!�� �5  �LM  �L`������
G
PROJ_GEMM_bwd(0:.���� �5  �LE @�NM  �L`��������
H
MHA_GEMM_2_bwd1(0B-��� �5  �LE  �LM  �N`��������
G
MHA_GEMM_2_bwd2(0B,��� �5  �LE  �NM  �L`�������
F
V_bwd	(0:5���� �5  �LE @�NM  �L`hu  �L��������
:
DropOut_1_bwd	(0J!�� �5  �NM  �N`������
8
SOFTMAX_bwd
(0J!�� �5  �NM  �N`������
G
MHA_GEMM_1_bwd1 (0B,��� �5  �NE  �LM  �L`�������
G
MHA_GEMM_1_bwd2!(0B,��� �5  �NE  �LM  �L`�������
?
Q_bwd"(0:.���� �5  �LE @�NM  �L`��������
?
K_bwd#(0:.���� �5  �LE @�NM  �L`��������
W
FFN1_bwd_weight_update$(0B5��� ��5  �LE  �MM @�O`hu @�O��d������
W
FFN0_bwd_weight_update%(0B5��� ��5  �ME  �LM @�O`hu @�O��������
\
PROJ_GEMM_bwd_weight_update&(0B5��� ��5  �LE  �LM @�N`hu @�N��������
T
V_bwd_weight_update'	(0B5��� ��5  �LE  �LM @�N`hu @�N��������
T
K_bwd_weight_update((0B5��� ��5  �LE  �LM @�N`hu @�N��������
T
Q_bwd_weight_update)(0B5��� ��5  �LE  �LM @�N`hu @�N��������4-   �0=  �LE  HKPZAdd_Prev_LayerbLayerNorm_1"0=  �LE  HKPZLayerNorm_1bQ"0=  �LE  HKPZLayerNorm_1bK"0=  �LE  HKPZLayerNorm_1bV!0=  �LE  �IPZQb
MHA_GEMM_1!0=  �LE  �IPZKb
MHA_GEMM_1'0=  �NE  �KPZ
MHA_GEMM_1bSOFTMAX&0=  �NE  �KPZSOFTMAXb	DropOut_1!	0=  �LE  �IP	ZVb
MHA_GEMM_2)	0=  �NE  �KP
Z	DropOut_1b
MHA_GEMM_2)	
0=  �LE  �IPZ
MHA_GEMM_2b	PROJ_GEMM-
-   �0=  �LE  HKPZ	PROJ_GEMMb	DropOut_2)-   �0=  �LE  HKPZ	DropOut_2bAdd_1.-   �0
=  �LE  HKPZAdd_Prev_LayerbAdd_1+-   �0=  �LE  HKPZAdd_1bLayerNorm_2%0=  �LE  HKPZLayerNorm_2bFFN0#-   �0=  �ME  �JPZFFN0bGeLU0=  �ME  �JPZGeLUbFFN1(-   �0=  �LE  HKPZFFN1b	DropOut_3$0=  �LE  HKPZ	DropOut_3bAdd_2 0=  �LE  HKPZAdd_1bAdd_2+0=  �LE  HKPZLoss_bwdbDropOut_3_bwd+0=  �LE  HKPZDropOut_3_bwdbFFN1_bwd+-   �0=  �ME  �JPZFFN1_bwdbGeLU_bwd&0=  �ME  �JPZGeLU_bwdbFFN0_bwd2-   �0=  �LE  HKPZFFN0_bwdbLayerNorm_2_bwd7-   �0=  �LE  HKPZLayerNorm_2_bwdbDropOut_2_bwd00=  �LE  HKPZDropOut_2_bwdbPROJ_GEMM_bwd20=  �LE  �IPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd1&0=  �LE  �IPZVbMHA_GEMM_2_bwd120=  �LE  �IPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd2.0=  �NE  �KP Z	DropOut_1bMHA_GEMM_2_bwd220=  �NE  �KP!ZMHA_GEMM_2_bwd1bDropOut_1_bwd*0=  �LE  �IP"ZMHA_GEMM_2_bwd2bV_bwd.0=  �NE  �KP#ZDropOut_1_bwdbSOFTMAX_bwd0 0=  �NE  �KP$ZSOFTMAX_bwdbMHA_GEMM_1_bwd10!0=  �NE  �KP%ZSOFTMAX_bwdbMHA_GEMM_1_bwd2& 0
=  �LE  �IP&ZKbMHA_GEMM_1_bwd1&!0
=  �LE  �IP'ZQbMHA_GEMM_1_bwd2* "0=  �LE  �IP(ZMHA_GEMM_1_bwd1bQ_bwd*!#0=  �LE  �IP)ZMHA_GEMM_1_bwd2bK_bwd9$0=  �LE  HKP*ZDropOut_3_bwdbFFN1_bwd_weight_update0$0=  �ME  �JP+ZGeLUbFFN1_bwd_weight_update4%0	=  �ME  �JP,ZGeLU_bwdbFFN0_bwd_weight_update7%0=  �LE  HKP-ZLayerNorm_2bFFN0_bwd_weight_update>&0=  �LE  HKP.ZDropOut_2_bwdbPROJ_GEMM_bwd_weight_update;	&0=  �LE  �IP/Z
MHA_GEMM_2bPROJ_GEMM_bwd_weight_update8'0=  �LE  �IP0ZMHA_GEMM_2_bwd2bV_bwd_weight_update9'-   �0	=  �LE  HKP1ZLayerNorm_1bV_bwd_weight_update8!(0=  �LE  �IP2ZMHA_GEMM_1_bwd2bK_bwd_weight_update4(0=  �LE  HKP3ZLayerNorm_1bK_bwd_weight_update8 )0=  �LE  �IP4ZMHA_GEMM_1_bwd1bQ_bwd_weight_update4)0=  �LE  HKP5ZLayerNorm_1bQ_bwd_weight_updateP���4 - TQ5�̌?b*��@��  �C�  HB�  HB�TP�PP�DP�
  @E  �Q2   @  �A  �?%��8J-�d*<5��T==��T=E��T=M�[f=U?��F"':���� �(�X�`h�px�fff?�*
o�:�