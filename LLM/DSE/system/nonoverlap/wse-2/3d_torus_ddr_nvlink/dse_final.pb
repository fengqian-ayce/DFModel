
�(
8
Add_Prev_Layer(0J �� �5  �LM  �L`�����
7
LayerNorm_1(0J �� �5  �LM  �L`�����
:
Q(0:-���� �5  �LE @�NM  �L`�������
:
K(0:-���� �5  �LE @�NM  �L`�������
:
V(0:-���� �5  �LE @�NM  �L`�������
B

MHA_GEMM_1(0B,��� �5  �LE  �LM  �N`�������
3
SOFTMAX(0J �� �5  �NM  �N`�����
5
	DropOut_1(0J �� �5  �NM  �N`�����
A

MHA_GEMM_2	(0B+��� �5  �LE  �NM  �L`������
I
	PROJ_GEMM
(0:4���� �5  �LE @�NM  �L`hu  �L�������
5
	DropOut_2(0J �� �5  �LM  �L`�����
7
Add_1	(0R&�� �5  �LM  �L`������  �L
7
LayerNorm_2
(0J �� �5  �LM  �L`�����
B
FFN0(0:2���� �5  �LE @�OM  �M`u   ���d�����
/
GeLU(0J�� �5  �MM  �M`��d��
D
FFN1(0:4���� �5  �ME @�OM  �L`hu  �L�����d��
5
	DropOut_3(0J �� �5  �LM  �L`�����
6
Add_2(0R%�� �5  �LM  �L`�����  �L
2
Loss_bwd(0J �� �5  �LM  �L`�����
9
DropOut_3_bwd(0J �� �5  �LM  �L`�����
A
FFN1_bwd(0:-���� �5  �LE @�OM  �M`��d�����
3
GeLU_bwd(0J�� �5  �MM  �M`��d��
H
FFN0_bwd(0:4���� �5  �ME @�OM  �L`hu  �L�����d��
;
LayerNorm_2_bwd(0J �� �5  �LM  �L`�����
9
DropOut_2_bwd(0J �� �5  �LM  �L`�����
F
PROJ_GEMM_bwd(0:-���� �5  �LE @�NM  �L`�������
G
MHA_GEMM_2_bwd1(0B,��� �5  �LE  �LM  �N`�������
F
MHA_GEMM_2_bwd2(0B+��� �5  �LE  �NM  �L`������
E
V_bwd	(0:4���� �5  �LE @�NM  �L`hu  �L�������
9
DropOut_1_bwd	(0J �� �5  �NM  �N`�����
7
SOFTMAX_bwd
(0J �� �5  �NM  �N`�����
F
MHA_GEMM_1_bwd1 (0B+��� �5  �NE  �LM  �L`������
F
MHA_GEMM_1_bwd2!(0B+��� �5  �NE  �LM  �L`������
>
Q_bwd"(0:-���� �5  �LE @�NM  �L`�������
>
K_bwd#(0:-���� �5  �LE @�NM  �L`�������
V
FFN1_bwd_weight_update$(0B4��� ��5  �LE  �MM @�O`hu @�O��d�����
V
FFN0_bwd_weight_update%(0B4��� ��5  �ME  �LM @�O`hu @�O�������
[
PROJ_GEMM_bwd_weight_update&(0B4��� ��5  �LE  �LM @�N`hu @�N�������
S
V_bwd_weight_update'	(0B4��� ��5  �LE  �LM @�N`hu @�N�������
S
K_bwd_weight_update((0B4��� ��5  �LE  �LM @�N`hu @�N�������
S
Q_bwd_weight_update)(0B4��� ��5  �LE  �LM @�N`hu @�N�������4-   �0=  �LE  HGPZAdd_Prev_LayerbLayerNorm_1"0=  �LE  HGPZLayerNorm_1bQ"0=  �LE  HGPZLayerNorm_1bK"0=  �LE  HGPZLayerNorm_1bV!0=  �LE  �EPZQb
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
MHA_GEMM_2bPROJ_GEMM_bwd_weight_update8'0=  �LE  �EP0ZMHA_GEMM_2_bwd2bV_bwd_weight_update9'-   �0	=  �LE  HGP1ZLayerNorm_1bV_bwd_weight_update8!(0=  �LE  �EP2ZMHA_GEMM_1_bwd2bK_bwd_weight_update4(0=  �LE  HGP3ZLayerNorm_1bK_bwd_weight_update8 )0=  �LE  �EP4ZMHA_GEMM_1_bwd1bQ_bwd_weight_update4)0=  �LE  HGP5ZLayerNorm_1bQ_bwd_weight_updateP���4 - TQ5�̌?b*��@��  �C�  HB�  HB�TP�PP�DP�
  �C  �S2   @  �A  �?%��8J-�d*<5��T==��T=E��T=M(a&>U?��F"':���� �(�X�`h�px�fff?�*
o�:�