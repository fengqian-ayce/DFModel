
Ώ
7
Add_Prev_Layer(0J 5   HM   H` 
6
LayerNorm_1(0J 5   HM   H` 
?
Q(0:2 5   HE   NM   H` ₯   L
?
K(0:2 5   HE   NM   H` ₯   L
?
V(0:2 5   HE   NM   H` ₯   L
G
K_cache(0J4  5   HM   M`  ₯   Kρ   Kρ   K
G
V_cache(0J4  5   HM   M`  ₯   Kρ   Kρ   K
@

MHA_GEMM_1(0B*  5   HE   MM   J`(  
1
SOFTMAX	(0J  5   JM   J`(  
3
	DropOut_1
(0J  5   JM   J`(  
@

MHA_GEMM_2(0B*  5   ME   JM   H`  
N
	PROJ_GEMM(0:9 5   HE   NM   H`hu   H ₯   L
4
	DropOut_2	(0J 5   HM   H` 
6
Add_1
(0R% 5   HM   H` υ   H
6
LayerNorm_2(0J 5   HM   H` 
B
FFN0(0:2  5   HE  ΠNM  I` ₯  ΠL
.
GeLU(0J  5  IM  I` 
I
FFN1(0:9  5  IE  ΠNM   H`hu   H ₯  ΠL
4
	DropOut_3(0J 5   HM   H` 
6
Add_2(0R% 5   HM   H` υ   H7-   0=   HE   HPZAdd_Prev_LayerbLayerNorm_1 %0=   HE   HPZLayerNorm_1bQ %0=   HE   HPZLayerNorm_1bK %0=   HE   HPZLayerNorm_1bV !0=   HE   FPZKbK_cache !0=   HE   FPZVbV_cache $0=   HE   FPZQb
MHA_GEMM_1 )0=   MPZK_cacheb
MHA_GEMM_1 ψπ*	0=   JE   HP	Z
MHA_GEMM_1bSOFTMAX )	
0=   JE   HP
ZSOFTMAXb	DropOut_1 ,
0=   JE   HPZ	DropOut_1b
MHA_GEMM_2 )0=   MPZV_cacheb
MHA_GEMM_2 ψπ,0=   HE   FPZ
MHA_GEMM_2b	PROJ_GEMM 0-   0=   HE   HPZ	PROJ_GEMMb	DropOut_2 ,-   0=   HE   HPZ	DropOut_2bAdd_1 .-   0=   HE   HPZAdd_1bLayerNorm_2 1-   0=   HE   HPZAdd_Prev_LayerbAdd_1 --   0=   HE   HPZLayerNorm_2bFFN0 &-   0=  IE  GPZFFN0bGeLU &-   0=  IE  GPZGeLUbFFN1 +-   0=   HE   HPZFFN1b	DropOut_3 ,-   0=   HE   HPZ	DropOut_3bAdd_2 (-   0=   HE   HPZAdd_1bAdd_2 8  -  NUΝΜΜ?½E  C" ­  HB²TP’
ΝΜΜD  Q(   @  ΐA  ?% θ G-Γd*<EτύT=M(a&>U ;D"+: (~X`xpxfff? *o:΄