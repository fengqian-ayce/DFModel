
�
9
Add_Prev_Layer(0J!�� �5  zNM  zN`������
9
LayerNorm_1 (0J �� �5  zNM  zN`����p�
=
Q (0:.���� �5  zNE $�QM  zN`�������T�
=
K (0:.���� �5  zNE $�QM  zN`�������R�
=
V (0:.���� �5  zNE $�QM  zN`�������R�
D

MHA_GEMM_1 (0B,��� �5  zNE  zNM  �O`���
���R�
5
SOFTMAX (0J �� �5  �OM  �O`���
�R�
7
	DropOut_1 (0J �� �5  �OM  �O`���
�S�
E

MHA_GEMM_2	 (0B-��� �5  zNE  �OM  zN`��������
L
	PROJ_GEMM
 	(0:5���� �5  zNE $�QM  zN`hu  zN�������R�
7
	DropOut_2 
(0J �� �5  zNM  zN`����R�
:
Add_1	 (0R'�� �5  zNM  zN`�������  zN
9
LayerNorm_2
 (0J �� �5  zNM  zN`����R�
@
FFN0 (0:.��>�� �5  zNE $�RM  zO`�������R�
2
GeLU (0J ��> �5  zOM  zO`����R�
G
FFN1 (0:5����> �5  zOE $�RM  zN`hu  zN�������R�
7
	DropOut_3 (0J �� �5  zNM  zN`����f�
:
Add_2 (0R'�� �5  zNM  zN`�������  zN7-   �0=  zNE ��LPZAdd_Prev_LayerbLayerNorm_1�%0=  zNEp�ZLPZLayerNorm_1bQ�%0=  zNEp�ZLPZLayerNorm_1bK�%0=  zNEp�ZLPZLayerNorm_1bV�$0=  zNE�#�JPZQb
MHA_GEMM_1�$0=  zNE�b�JPZKb
MHA_GEMM_1�*0=  �OE  �KPZ
MHA_GEMM_1bSOFTMAX�)0=  �OE  �KPZSOFTMAXb	DropOut_1�$	0=  zNE (�JP	ZVb
MHA_GEMM_2�,	0=  �OE�[�KP
Z	DropOut_1b
MHA_GEMM_2�,	
0=  zNE XKPZ
MHA_GEMM_2b	PROJ_GEMM�0
-   �0=  zNE ( LPZ	PROJ_GEMMb	DropOut_2�,-   �0=  zNE ( LPZ	DropOut_2bAdd_1�1-   �0
=  zNE ��LPZAdd_Prev_LayerbAdd_1�.-   �0=  zNE ��LPZAdd_1bLayerNorm_2�--   �0=  zNE ( LPZLayerNorm_2bFFN0�&-   �0=  zOE (�KPZFFN0bGeLU�&-   �0=  zOE (�KPZGeLUbFFN1�+-   �0=  zNE ( LPZFFN1b	DropOut_3�,-   �0=  zNE 8GLPZ	DropOut_3bAdd_2�(-   �0=  zNE ��LPZAdd_1bAdd_2�-�  -  )NU���?"��  zD�TP�  �D2   @  �A  �?%��F-��T=5��T==��T=E��T=M(a&>UeZ�C"":���� �(�X��`xp��*o�:�