
Ë
7
Add_Prev_Layer(0J` 5  @LM  @L``ó 
6
LayerNorm_1(0J` 5  @LM  @L``Ù 
7
Q(0:*`` 5  @LE  MM  @L`by 
7
K(0:*`` 5  @LE  MM  @L``z 
7
V(0:*`` 5  @LE  MM  @L``y 
A

MHA_GEMM_1(0B+` 5  @LE  @LM  @N`y 
2
SOFTMAX(0J` 5  @NM  @N`y 
6
	DropOut_1 (0J` 5  @NM  @N`y 
B

MHA_GEMM_2	 (0B*` 5  @LE  @NM  @L`y 
H
	PROJ_GEMM
 (0:1`` 5  @LE  MM  @L`hu  @L`y 
6
	DropOut_2 (0J` 5  @LM  @L`` 
7
Add_1	 (0R$` 5  @LM  @L``y õ  @L
7
LayerNorm_2
 (0J` 5  @LM  @L``y 
B
FFN0 (0:0` 5  @LE  NM  @M`u   ``y 
1
GeLU (0J 5  @MM  @M``y 
D
FFN1 (0:2` 5  @ME  NM  @L`hu  @L``y 
6
	DropOut_3 (0J` 5  @LM  @L`aä  
7
Add_2 (0R$` 5  @LM  @L``y õ  @L7-   0=  @LE  ;KPZAdd_Prev_LayerbLayerNorm_1 %0=  @LE `KPZLayerNorm_1bQ %0=  @LE `KPZLayerNorm_1bK %0=  @LE `KPZLayerNorm_1bV $0=  @LE 5IPZQb
MHA_GEMM_1 $0=  @LE  7IPZKb
MHA_GEMM_1 *0=  @NE 5KPZ
MHA_GEMM_1bSOFTMAX )0=  @NE 5KPZSOFTMAXb	DropOut_1 $	0=  @LE 5IP	ZVb
MHA_GEMM_2 ,	0=  @NE 5KP
Z	DropOut_1b
MHA_GEMM_2 ,	
0=  @LE 5IPZ
MHA_GEMM_2b	PROJ_GEMM 0
-   0=  @LE 5JPZ	PROJ_GEMMb	DropOut_2 ,-   0=  @LE ÀÕJPZ	DropOut_2bAdd_1 1-   0
=  @LE  ;KPZAdd_Prev_LayerbAdd_1 .-   0=  @LE 5JPZAdd_1bLayerNorm_2 --   0=  @LE 5JPZLayerNorm_2bFFN0 &-   0=  @ME 5JPZFFN0bGeLU &-   0=  @ME 5JPZGeLUbFFN1 +-   0=  @LE 5JPZFFN1b	DropOut_3 ,-   0=  @LEn7ÇLPZ	DropOut_3bAdd_2 (-   0=  @LE 5JPZAdd_1bAdd_2 @  -  úMU   ?: ¨µ  HB½  HBÂTPÊDP¢
  C  S2   @  ÀA  ?%F-ôýT=5ôýT==ôýT=EôýT=M(a&>UeZÞC" :`` (`X`xpx*o:x