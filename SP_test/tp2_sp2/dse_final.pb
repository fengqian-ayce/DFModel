
ί
6
Add_Prev_Layer(0J` 5  @LM  @L``l 
7
LayerNorm_1 (0J` 5  @LM  @L``o 
9
Q (0:*`` 5  @LE  MM  @L`0`l 
9
K (0:*`` 5  @LE  MM  @L`0`l 
9
V (0:*`` 5  @LE  MM  @L`0`r 
C

MHA_GEMM_1 (0B+` 5  @LE  @LM  @N`Ύ	r 
4
SOFTMAX (0J` 5  @NM  @N`l 
6
	DropOut_1 (0J` 5  @NM  @N`r 
E

MHA_GEMM_2	 (0B-` 5  @LE  @NM  @L`0r ΐ
H
	PROJ_GEMM
 (0:1`` 5  @LE  MM  @L`hu  @L`0l 
6
	DropOut_2 (0J` 5  @LM  @L`έl 
8
Add_1	 (0R%` 5  @LM  @L``μ υ  @L
8
LayerNorm_2
 (0J` 5  @LM  @L``― 
C
FFN0 (0:1` 5  @LE  NM  @M`u   ΐ`l 
2
GeLU (0J  5  @MM  @M`ΐl 
E
FFN1 (0:3` 5  @ME  NM  @L`hu  @L`ΐr 
5
	DropOut_3 (0J` 5  @LM  @L``l 
7
Add_2 (0R$` 5  @LM  @L``l υ  @L7-   0=  @LE  "JPZAdd_Prev_LayerbLayerNorm_1 %0=  @LE &JPZLayerNorm_1bQ %0=  @LE &JPZLayerNorm_1bK %0=  @LE &JPZLayerNorm_1bV $0=  @LE  ’IPZQb
MHA_GEMM_1 '0=  @LE  ’IPZKb
MHA_GEMM_1 Έ>*0=  @NE  «KPZ
MHA_GEMM_1bSOFTMAX )0=  @NE  ’KPZSOFTMAXb	DropOut_1 $	0=  @LE  «IP	ZVb
MHA_GEMM_2 ,	0=  @NE  «KP
Z	DropOut_1b
MHA_GEMM_2 ,	
0=  @LE  «IPZ
MHA_GEMM_2b	PROJ_GEMM 0
-   0=  @LE  "JPZ	PROJ_GEMMb	DropOut_2 ,-   0=  @LEΈMKPZ	DropOut_2bAdd_1 1-   0
=  @LE  "JPZAdd_Prev_LayerbAdd_1 .-   0=  @LE KPZAdd_1bLayerNorm_2 --   0=  @LE @JPZLayerNorm_2bFFN0 &-   0=  @ME  ’JPZFFN0bGeLU &-   0=  @ME  ’JPZGeLUbFFN1 +-   0=  @LE  +JPZFFN1b	DropOut_3 ,-   0=  @LE  "JPZ	DropOut_3bAdd_2 (-   0=  @LE KPZAdd_1bAdd_2 \  -   MU   ?z8 ¨°ΈΕ  HCΝ  HCΥ  HCέ  HCβSPκTPςPPϊDP’
  ΘB  S2   @  ΐA  ?%F-τύT=5τύT==τύT=EτύT=M(a&>UeZήC" :`` (`X`xpx
*o:΄