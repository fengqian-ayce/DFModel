

5
Add_Prev_Layer(0J  5   GM   G`  
4
LayerNorm_1(0J  5   GM   G`  
<
Q(0:/   5   GE   LM   G`  ₯   J
<
K(0:/   5   GE   LM   G`  ₯   J
<
V(0:/   5   GE   LM   G`  ₯   J
F
K_cache(0J3   5   GM   L`  ₯   Jρ   Jρ   J
F
V_cache(0J3   5   GM   L`  ₯   Jρ   Jρ   J
?

MHA_GEMM_1(0B)   5   GE   LM   I`
  
0
SOFTMAX	(0J   5   IM   I`
  
2
	DropOut_1
(0J   5   IM   I`
  
?

MHA_GEMM_2(0B)   5   LE   IM   G`  
K
	PROJ_GEMM(0:6   5   GE   LM   G`hu   G  ₯   J
2
	DropOut_2	(0J  5   GM   G`  
4
Add_1
(0R#  5   GM   G`  υ   G
4
LayerNorm_2(0J  5   GM   G`  
?
FFN0(0:/p  5   GE  ΰLM  H`  ₯  ΰJ
-
GeLU(0Jp 5  HM  H` 
F
FFN1(0:6 p 5  HE  ΰLM   G`hu   G  ₯  ΰJ
2
	DropOut_3(0J  5   GM   G`  
4
Add_2(0R#  5   GM   G`  υ   G7-   0=   GE   GPZAdd_Prev_LayerbLayerNorm_1 %0=   GE   GPZLayerNorm_1bQ %0=   GE   GPZLayerNorm_1bK %0=   GE   GPZLayerNorm_1bV !0=   GE   EPZKbK_cache !0=   GE   EPZVbV_cache $0=   GE   EPZQb
MHA_GEMM_1 )0=   LPZK_cacheb
MHA_GEMM_1 ψπ*	0=   IE   GP	Z
MHA_GEMM_1bSOFTMAX )	
0=   IE   GP
ZSOFTMAXb	DropOut_1 ,
0=   IE   GPZ	DropOut_1b
MHA_GEMM_2 )0=   LPZV_cacheb
MHA_GEMM_2 ψπ,0=   GE   EPZ
MHA_GEMM_2b	PROJ_GEMM 0-   0=   GE   GPZ	PROJ_GEMMb	DropOut_2 ,-   0=   GE   GPZ	DropOut_2bAdd_1 .-   0=   GE   GPZAdd_1bLayerNorm_2 1-   0=   GE   GPZAdd_Prev_LayerbAdd_1 --   0=   GE   GPZLayerNorm_2bFFN0 &-   0=  HE  FPZFFN0bGeLU &-   0=  HE  FPZGeLUbFFN1 +-   0=   GE   GPZFFN1b	DropOut_3 ,-   0=   GE   GPZ	DropOut_3bAdd_2 (-   0=   GE   GPZAdd_1bAdd_2 8  -  NUΝΜΜ?½E  C" ­  HB²TP’
ΝΜΜD  Q(   @  ΐA  ?% θ G-Γd*<EτύT=M(a&>U ;D"):   ( X`xpxfff? *o:΄