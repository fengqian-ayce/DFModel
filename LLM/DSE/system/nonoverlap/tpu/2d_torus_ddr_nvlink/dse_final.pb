
�)
8
Add_Prev_Layer(0J �� �5  �LM  �L`�����
9
LayerNorm_1 (0J �� �5  �LM  �L`�����
<
Q (0:-���� �5  �LE @�NM  �L`�������
<
K (0:-���� �5  �LE @�NM  �L`�������
<
V (0:-���� �5  �LE @�NM  �L`�������
D

MHA_GEMM_1 (0B,��� �5  �LE  �LM  �N`�������
5
SOFTMAX (0J �� �5  �NM  �N`�����
7
	DropOut_1 (0J �� �5  �NM  �N`�����
C

MHA_GEMM_2	 (0B+��� �5  �LE  �NM  �L`������
K
	PROJ_GEMM
 	(0:4���� �5  �LE @�NM  �L`hu  �L�������
7
	DropOut_2 
(0J �� �5  �LM  �L`�����
9
Add_1	 (0R&�� �5  �LM  �L`������  �L
9
LayerNorm_2
 (0J �� �5  �LM  �L`�����
D
FFN0 (0:2���� �5  �LE @�OM  �M`u   ���d�����
1
GeLU (0J�� �5  �MM  �M`��d��
F
FFN1 (0:4���� �5  �ME @�OM  �L`hu  �L�����d��
7
	DropOut_3 (0J �� �5  �LM  �L`�����
8
Add_2 (0R%�� �5  �LM  �L`�����  �L
4
Loss_bwd (0J �� �5  �LM  �L`�����
;
DropOut_3_bwd (0J �� �5  �LM  �L`�����
C
FFN1_bwd (0:-���� �5  �LE @�OM  �M`��d�����
5
GeLU_bwd (0J�� �5  �MM  �M`��d��
J
FFN0_bwd (0:4���� �5  �ME @�OM  �L`hu  �L�����d��
=
LayerNorm_2_bwd (0J �� �5  �LM  �L`�����
;
DropOut_2_bwd (0J �� �5  �LM  �L`�����
H
PROJ_GEMM_bwd (0:-���� �5  �LE @�NM  �L`�������
I
MHA_GEMM_2_bwd1 (0B,��� �5  �LE  �LM  �N`�������
H
MHA_GEMM_2_bwd2 (0B+��� �5  �LE  �NM  �L`������
G
V_bwd	 (0:4���� �5  �LE @�NM  �L`hu  �L�������
;
DropOut_1_bwd	 (0J �� �5  �NM  �N`�����
9
SOFTMAX_bwd
 (0J �� �5  �NM  �N`�����
H
MHA_GEMM_1_bwd1  (0B+��� �5  �NE  �LM  �L`������
H
MHA_GEMM_1_bwd2!  (0B+��� �5  �NE  �LM  �L`������
@
Q_bwd" !(0:-���� �5  �LE @�NM  �L`�������
@
K_bwd# "(0:-���� �5  �LE @�NM  �L`�������
X
FFN1_bwd_weight_update$ #(0B4��� ��5  �LE  �MM @�O`hu @�O��d�����
X
FFN0_bwd_weight_update% $(0B4��� ��5  �ME  �LM @�O`hu @�O�������
]
PROJ_GEMM_bwd_weight_update& %(0B4��� ��5  �LE  �LM @�N`hu @�N�������
U
V_bwd_weight_update'	 &(0B4��� ��5  �LE  �LM @�N`hu @�N�������
U
K_bwd_weight_update( '(0B4��� ��5  �LE  �LM @�N`hu @�N�������
U
Q_bwd_weight_update) ((0B4��� ��5  �LE  �LM @�N`hu @�N�������4-   �0=  �LE  �GPZAdd_Prev_LayerbLayerNorm_1"0=  �LE  �GPZLayerNorm_1bQ"0=  �LE  �GPZLayerNorm_1bK"0=  �LE  �GPZLayerNorm_1bV!0=  �LE  HFPZQb
MHA_GEMM_1!0=  �LE  HFPZKb
MHA_GEMM_1'0=  �NE   HPZ
MHA_GEMM_1bSOFTMAX&0=  �NE   HPZSOFTMAXb	DropOut_1!	0=  �LE  HFP	ZVb
MHA_GEMM_2)	0=  �NE   HP
Z	DropOut_1b
MHA_GEMM_2)	
0=  �LE  HFPZ
MHA_GEMM_2b	PROJ_GEMM-
-   �0=  �LE  �GPZ	PROJ_GEMMb	DropOut_2)-   �0=  �LE  �GPZ	DropOut_2bAdd_1.-   �0
=  �LE  �GPZAdd_Prev_LayerbAdd_1+-   �0=  �LE  �GPZAdd_1bLayerNorm_2%0=  �LE  �GPZLayerNorm_2bFFN0#-   �0=  �ME  HGPZFFN0bGeLU0=  �ME  HGPZGeLUbFFN1(-   �0=  �LE  �GPZFFN1b	DropOut_3$0=  �LE  �GPZ	DropOut_3bAdd_2 0=  �LE  �GPZAdd_1bAdd_2+0=  �LE  �GPZLoss_bwdbDropOut_3_bwd+0=  �LE  �GPZDropOut_3_bwdbFFN1_bwd+-   �0=  �ME  HGPZFFN1_bwdbGeLU_bwd&0=  �ME  HGPZGeLU_bwdbFFN0_bwd2-   �0=  �LE  �GPZFFN0_bwdbLayerNorm_2_bwd7-   �0=  �LE  �GPZLayerNorm_2_bwdbDropOut_2_bwd00=  �LE  �GPZDropOut_2_bwdbPROJ_GEMM_bwd20=  �LE  HFPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd1&0=  �LE  HFPZVbMHA_GEMM_2_bwd120=  �LE  HFPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd2.0=  �NE   HP Z	DropOut_1bMHA_GEMM_2_bwd220=  �NE   HP!ZMHA_GEMM_2_bwd1bDropOut_1_bwd*0=  �LE  HFP"ZMHA_GEMM_2_bwd2bV_bwd.0=  �NE   HP#ZDropOut_1_bwdbSOFTMAX_bwd0 0=  �NE   HP$ZSOFTMAX_bwdbMHA_GEMM_1_bwd10!0=  �NE   HP%ZSOFTMAX_bwdbMHA_GEMM_1_bwd2& 0
=  �LE  HFP&ZKbMHA_GEMM_1_bwd1&!0
=  �LE  HFP'ZQbMHA_GEMM_1_bwd2* "0=  �LE  HFP(ZMHA_GEMM_1_bwd1bQ_bwd*!#0=  �LE  HFP)ZMHA_GEMM_1_bwd2bK_bwd9$0=  �LE  �GP*ZDropOut_3_bwdbFFN1_bwd_weight_update0$0=  �ME  HGP+ZGeLUbFFN1_bwd_weight_update4%0	=  �ME  HGP,ZGeLU_bwdbFFN0_bwd_weight_update7%0=  �LE  �GP-ZLayerNorm_2bFFN0_bwd_weight_update>&0=  �LE  �GP.ZDropOut_2_bwdbPROJ_GEMM_bwd_weight_update;	&0=  �LE  HFP/Z
MHA_GEMM_2bPROJ_GEMM_bwd_weight_update8'0=  �LE  HFP0ZMHA_GEMM_2_bwd2bV_bwd_weight_update9'-   �0	=  �LE  �GP1ZLayerNorm_1bV_bwd_weight_update8!(0=  �LE  HFP2ZMHA_GEMM_1_bwd2bK_bwd_weight_update4(0=  �LE  �GP3ZLayerNorm_1bK_bwd_weight_update8 )0=  �LE  HFP4ZMHA_GEMM_1_bwd1bQ_bwd_weight_update4)0=  �LE  �GP5ZLayerNorm_1bQ_bwd_weight_updateC�� �-   M5ff�?:����  �C�  HB�TP�PP�
  �C  �S-   @  �A  �?%�֋E-�d*<5��T=E��T=M(a&>U��*C"%:���� �(�X�`h�p�fff?�*
o�:�