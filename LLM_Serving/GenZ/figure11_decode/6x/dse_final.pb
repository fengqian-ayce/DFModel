

5
Add_Prev_Layer(0J` 5  @FM  @F`` 
4
LayerNorm_1(0J` 5  @FM  @F`` 
<
Q(0:/`` 5  @FE  MM  @F`` ₯  @K
<
K(0:/`` 5  @FE  MM  @F`` ₯  @K
<
V(0:/`` 5  @FE  MM  @F`` ₯  @K
8
K_cache(0J%` u5  @FM Λ/M`u ₯ `jK
8
V_cache(0J%` u5  @FM Λ/M`u ₯ `jK
?

MHA_GEMM_1(0B)` u5  @FE Λ/MM Λ―I`u 
0
SOFTMAX	(0J` u5 Λ―IM Λ―I`u 
2
	DropOut_1
(0J` u5 Λ―IM Λ―I`u 
?

MHA_GEMM_2(0B)`u 5 Λ/ME Λ―IM  @F`u 
K
	PROJ_GEMM(0:6`` 5  @FE  MM  @F`hu  @F` ₯  @K
2
	DropOut_2	(0J` 5  @FM  @F`` 
4
Add_1
(0R#` 5  @FM  @F`` υ  @F
4
LayerNorm_2(0J` 5  @FM  @F`` 
@
FFN0(0:0` 5  @FE  NM  @G` ` ₯  @L
.
GeLU(0J 5  @GM  @G`  
G
FFN1(0:7` 5  @GE  NM  @F`hu  @F`  ₯  @L
2
	DropOut_3(0J` 5  @FM  @F`` 
4
Add_2(0R#` 5  @FM  @F`` υ  @F7-   0=  @FE  @FPZAdd_Prev_LayerbLayerNorm_1 %0=  @FE  @FPZLayerNorm_1bQ %0=  @FE  @FPZLayerNorm_1bK %0=  @FE  @FPZLayerNorm_1bV !0=  @FE  DPZKbK_cache !0=  @FE  DPZVbV_cache $0=  @FE  DPZQb
MHA_GEMM_1 *0= Λ/ME djKPZK_cacheb
MHA_GEMM_1 *	0= Λ―IE dκGP	Z
MHA_GEMM_1bSOFTMAX )	
0= Λ―IE dκGP
ZSOFTMAXb	DropOut_1 ,
0= Λ―IE dκGPZ	DropOut_1b
MHA_GEMM_2 *0= Λ/ME djKPZV_cacheb
MHA_GEMM_2 ,0=  @FE  DPZ
MHA_GEMM_2b	PROJ_GEMM 0-   0=  @FE  @FPZ	PROJ_GEMMb	DropOut_2 ,-   0=  @FE  @FPZ	DropOut_2bAdd_1 .-   0=  @FE  @FPZAdd_1bLayerNorm_2 1-   0=  @FE  @FPZAdd_Prev_LayerbAdd_1 --   0=  @FE  @FPZLayerNorm_2bFFN0 &-   0=  @GE  EPZFFN0bGeLU &-   0=  @GE  EPZGeLUbFFN1 +-   0=  @FE  @FPZFFN1b	DropOut_3 ,-   0=  @FE  @FPZ	DropOut_3bAdd_2 (-   0=  @FE  @FPZAdd_1bAdd_2 8΅ -  ΘQU  ?½E  ϊD" ­  C²TP’
 »F   Q(   @  ΐA  ?% θ G-Γd*<EτύT=M(a&>U ;D"):`` (`X`hxpx  ? *o:΄