
�
8
Add_Prev_Layer(0J �� �5  zNM  zN`�����
9
LayerNorm_1 (0J �� �5  zNM  zN`�����
=
Q (0:.���� �5  zNE $�QM  zN`��������
=
K (0:.���� �5  zNE $�QM  zN`��������
=
V (0:.���� �5  zNE $�QM  zN`��������
D

MHA_GEMM_1 (0B,��� �5  zNE  zNM  �O`���
����
5
SOFTMAX (0J �� �5  �OM  �O`���
��
7
	DropOut_1 (0J �� �5  �OM  �O`���
��
D

MHA_GEMM_2	 (0B,��� �5  zNE  �OM  zN`�������
L
	PROJ_GEMM
 	(0:5���� �5  zNE $�QM  zN`hu  zN��������
7
	DropOut_2 
(0J �� �5  zNM  zN`�����
9
Add_1	 (0R&�� �5  zNM  zN`������  zN
9
LayerNorm_2
 (0J �� �5  zNM  zN`�����
@
FFN0 (0:.��>�� �5  zNE $�RM  zO`��������
2
GeLU (0J ��> �5  zOM  zO`�����
G
FFN1 (0:5����> �5  zOE $�RM  zN`hu  zN��������
7
	DropOut_3 (0J �� �5  zNM  zN`�����
9
Add_2 (0R&�� �5  zNM  zN`������  zN7-   �0=  zNE AJPZAdd_Prev_LayerbLayerNorm_1�%0=  zNEp�AJPZLayerNorm_1bQ�%0=  zNEp�AJPZLayerNorm_1bK�%0=  zNEp�AJPZLayerNorm_1bV�$0=  zNE���HPZQb
MHA_GEMM_1�$0=  zNE ��HPZKb
MHA_GEMM_1�*0=  �OE  �IPZ
MHA_GEMM_1bSOFTMAX�)0=  �OE  �IPZSOFTMAXb	DropOut_1�$	0=  zNE���HP	ZVb
MHA_GEMM_2�,	0=  �OE  �IP
Z	DropOut_1b
MHA_GEMM_2�,	
0=  zNE ��HPZ
MHA_GEMM_2b	PROJ_GEMM�0
-   �0=  zNE �;JPZ	PROJ_GEMMb	DropOut_2�,-   �0=  zNE �;JPZ	DropOut_2bAdd_1�1-   �0
=  zNE AJPZAdd_Prev_LayerbAdd_1�.-   �0=  zNE �;JPZAdd_1bLayerNorm_2�--   �0=  zNE �;JPZLayerNorm_2bFFN0�&-   �0=  zOE ��IPZFFN0bGeLU�&-   �0=  zOE ��IPZGeLUbFFN1�+-   �0=  zNE �;JPZFFN1b	DropOut_3�,-   �0=  zNE �;JPZ	DropOut_3bAdd_2�(-   �0=  zNE �;JPZAdd_1bAdd_2�-�
  -  �MU���?"��  zD�TP�  �D2   @  �A  �?%��F-��T=5��T==��T=E��T=M(a&>UeZ�C"":���� �(�X��`xp��*o�:�