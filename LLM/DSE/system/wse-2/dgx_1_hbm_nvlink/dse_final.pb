
�'
5
Add_Prev_Layer(0J�� �5  �LM  �L`����
4
LayerNorm_1(0J�� �5  �LM  �L`����
:
Q(0:-���� �5  �LE @�NM  �L`�������
:
K(0:-���� �5  �LE @�NM  �L`�������
:
V(0:-���� �5  �LE @�NM  �L`�������
B

MHA_GEMM_1(0B,��� �5  �LE  �LM  �N`�������
0
SOFTMAX(0J�� �5  �NM  �N`����
2
	DropOut_1(0J�� �5  �NM  �N`����
>

MHA_GEMM_2	(0B(��� �5  �LE  �NM  �L`�����
I
	PROJ_GEMM
(0:4���� �5  �LE @�NM  �L`hu  �L�������
2
	DropOut_2(0J�� �5  �LM  �L`����
4
Add_1	(0R#�� �5  �LM  �L`�����  �L
4
LayerNorm_2
(0J�� �5  �LM  �L`����
B
FFN0(0:2���� �5  �LE @�OM  �M`u   ���d�����
,
GeLU(0J�� �5  �MM  �M`��d�
A
FFN1(0:1���� �5  �ME @�OM  �L`hu  �L�����d�
2
	DropOut_3(0J�� �5  �LM  �L`����
6
Add_2(0R%�� �5  �LM  �L`�����  �L
/
Loss_bwd(0J�� �5  �LM  �L`����
6
DropOut_3_bwd(0J�� �5  �LM  �L`����
A
FFN1_bwd(0:-���� �5  �LE @�OM  �M`��d�����
0
GeLU_bwd(0J�� �5  �MM  �M`��d�
H
FFN0_bwd(0:4���� �5  �ME @�OM  �L`hu  �L�����d��
8
LayerNorm_2_bwd(0J�� �5  �LM  �L`����
6
DropOut_2_bwd(0J�� �5  �LM  �L`����
F
PROJ_GEMM_bwd(0:-���� �5  �LE @�NM  �L`�������
G
MHA_GEMM_2_bwd1(0B,��� �5  �LE  �LM  �N`�������
C
MHA_GEMM_2_bwd2(0B(��� �5  �LE  �NM  �L`�����
B
V_bwd	(0:1���� �5  �LE @�NM  �L`hu  �L������
6
DropOut_1_bwd	(0J�� �5  �NM  �N`����
4
SOFTMAX_bwd
(0J�� �5  �NM  �N`����
C
MHA_GEMM_1_bwd1 (0B(��� �5  �NE  �LM  �L`�����
C
MHA_GEMM_1_bwd2!(0B(��� �5  �NE  �LM  �L`�����
;
Q_bwd"(0:*���� �5  �LE @�NM  �L`������
;
K_bwd#(0:*���� �5  �LE @�NM  �L`������
S
FFN1_bwd_weight_update$(0B1��� ��5  �LE  �MM @�O`hu @�O��d����
S
FFN0_bwd_weight_update%(0B1��� ��5  �ME  �LM @�O`hu @�O������
[
PROJ_GEMM_bwd_weight_update&(0B4��� ��5  �LE  �LM @�N`hu @�N�������
S
V_bwd_weight_update'	(0B4��� ��5  �LE  �LM @�N`hu @�N�������
P
K_bwd_weight_update((0B1��� ��5  �LE  �LM @�N`hu @�N������
P
Q_bwd_weight_update)(0B1��� ��5  �LE  �LM @�N`hu @�N������4-   �0=  �LE  HGPZAdd_Prev_LayerbLayerNorm_1"0=  �LE  HGPZLayerNorm_1bQ"0=  �LE  HGPZLayerNorm_1bK"0=  �LE  HGPZLayerNorm_1bV!0=  �LE  �EPZQb
MHA_GEMM_1!0=  �LE  �EPZKb
MHA_GEMM_1'0=  �NE  �GPZ
MHA_GEMM_1bSOFTMAX&0=  �NE  �GPZSOFTMAXb	DropOut_1!	0=  �LE  �EP	ZVb
MHA_GEMM_2)	0=  �NE  �GP
Z	DropOut_1b
MHA_GEMM_2)	
0=  �LE  �EPZ
MHA_GEMM_2b	PROJ_GEMM-
-   �0=  �LE  HGPZ	PROJ_GEMMb	DropOut_2)-   �0=  �LE  HGPZ	DropOut_2bAdd_1.-   �0
=  �LE  HGPZAdd_Prev_LayerbAdd_1+-   �0=  �LE  HGPZAdd_1bLayerNorm_2%0=  �LE  HGPZLayerNorm_2bFFN0#-   �0=  �ME  �FPZFFN0bGeLU0=  �ME  �FPZGeLUbFFN1(-   �0=  �LE  HGPZFFN1b	DropOut_3$0=  �LE  HGPZ	DropOut_3bAdd_2 0=  �LE  HGPZAdd_1bAdd_2+0=  �LE  HGPZLoss_bwdbDropOut_3_bwd+0=  �LE  HGPZDropOut_3_bwdbFFN1_bwd+-   �0=  �ME  �FPZFFN1_bwdbGeLU_bwd&0=  �ME  �FPZGeLU_bwdbFFN0_bwd2-   �0=  �LE  HGPZFFN0_bwdbLayerNorm_2_bwd7-   �0=  �LE  HGPZLayerNorm_2_bwdbDropOut_2_bwd00=  �LE  HGPZDropOut_2_bwdbPROJ_GEMM_bwd20=  �LE  �EPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd1&0=  �LE  �EPZVbMHA_GEMM_2_bwd120=  �LE  �EPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd2.0=  �NE  �GP Z	DropOut_1bMHA_GEMM_2_bwd220=  �NE  �GP!ZMHA_GEMM_2_bwd1bDropOut_1_bwd*0=  �LE  �EP"ZMHA_GEMM_2_bwd2bV_bwd.0=  �NE  �GP#ZDropOut_1_bwdbSOFTMAX_bwd0 0=  �NE  �GP$ZSOFTMAX_bwdbMHA_GEMM_1_bwd10!0=  �NE  �GP%ZSOFTMAX_bwdbMHA_GEMM_1_bwd2& 0
=  �LE  �EP&ZKbMHA_GEMM_1_bwd1&!0
=  �LE  �EP'ZQbMHA_GEMM_1_bwd2* "0=  �LE  �EP(ZMHA_GEMM_1_bwd1bQ_bwd*!#0=  �LE  �EP)ZMHA_GEMM_1_bwd2bK_bwd9$0=  �LE  HGP*ZDropOut_3_bwdbFFN1_bwd_weight_update0$0=  �ME  �FP+ZGeLUbFFN1_bwd_weight_update4%0	=  �ME  �FP,ZGeLU_bwdbFFN0_bwd_weight_update7%0=  �LE  HGP-ZLayerNorm_2bFFN0_bwd_weight_update>&0=  �LE  HGP.ZDropOut_2_bwdbPROJ_GEMM_bwd_weight_update;	&0=  �LE  �EP/Z
MHA_GEMM_2bPROJ_GEMM_bwd_weight_update8'0=  �LE  �EP0ZMHA_GEMM_2_bwd2bV_bwd_weight_update9'-   �0	=  �LE  HGP1ZLayerNorm_1bV_bwd_weight_update8!(0=  �LE  �EP2ZMHA_GEMM_1_bwd2bK_bwd_weight_update4(0=  �LE  HGP3ZLayerNorm_1bK_bwd_weight_update8 )0=  �LE  �EP4ZMHA_GEMM_1_bwd1bQ_bwd_weight_update4)0=  �LE  HGP5ZLayerNorm_1bQ_bwd_weight_updateC���4 - TQ5�̌?J����  �C�  HB�TP�PP�
  @E  �Q-   @  �A  �?%��8J-�d*<5��T=E��T=M�[f=U?��F"*:���� �(�X�`h�px��fff?�*
o�:�