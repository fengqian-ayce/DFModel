
�
:
Add_Prev_Layer ���������(0J�@ � 5  �LM  �L`�
9
LayerNorm_1 ���������(0J�@ � 5  �LM  �L`�
7
Q ���������(0:@��@ � 5  �LE   MM  �L`�
7
K ���������(0:@��@ � 5  �LE   MM  �L`�
7
V ���������(0:@��@ � 5  �LE   MM  �L`�
@

MHA_GEMM_1 ���������(0B@� � � 5  �LE  �LM   O`�
5
SOFTMAX ���������(0J@�  � 5   OM   O`�
7
	DropOut_1 ���������(0J@�  � 5   OM   O`�
@

MHA_GEMM_2	 ���������(0B@��  � 5  �LE   OM  �L`�
F
	PROJ_GEMM
 ���������(0:&�@�@ � 5  �LE   MM  �L`hu  �L�
7
	DropOut_2 ���������(0J�@ � 5  �LM  �L`�
9
Add_1	 ���������(0R�@ � 5  �LM  �L`��  �L
9
LayerNorm_2
 ���������(0J�@ � 5  �LM  �L`�
@
FFN0 ���������(0:%���@ � 5  �LE  �MM  `M`u   ��
3
GeLU ���������(0J�� � 5  `MM  `M`�
B
FFN1 ���������(0:'�@�� � 5  `ME  �MM  �L`hu  �L�
7
	DropOut_3 ���������(0J�@ � 5  �LM  �L`�
9
Add_2 ���������(0R�@ � 5  �LM  �L`��  �L2-   �0=  �LPZAdd_Prev_LayerbLayerNorm_1� 0=  �LPZLayerNorm_1bQ� 0=  �LPZLayerNorm_1bK� 0=  �LPZLayerNorm_1bV�0=  �LPZQb
MHA_GEMM_1�0=  �LPZKb
MHA_GEMM_1�%0=   OPZ
MHA_GEMM_1bSOFTMAX�$0=   OPZSOFTMAXb	DropOut_1�	0=  �LP	ZVb
MHA_GEMM_2�'	0=   OP
Z	DropOut_1b
MHA_GEMM_2�'	
0=  �LPZ
MHA_GEMM_2b	PROJ_GEMM�+
-   �0=  �LPZ	PROJ_GEMMb	DropOut_2�'-   �0=  �LPZ	DropOut_2bAdd_1�,-   �0
=  �LPZAdd_Prev_LayerbAdd_1�)-   �0=  �LPZAdd_1bLayerNorm_2�(-   �0=  �LPZLayerNorm_2bFFN0�!-   �0=  `MPZFFN0bGeLU�!-   �0=  `MPZGeLUbFFN1�&-   �0=  �LPZFFN1b	DropOut_3�'-   �0=  �LPZ	DropOut_3bAdd_2�#-   �0=  �LPZAdd_1bAdd_2�2� -  �LU�"�?*��  aD�TP�
 ��D  �Q(   @  �A  �?% � G-�d*<E��T=M(a&>U �;D" :�@@� � (PX`xp���*o�:�