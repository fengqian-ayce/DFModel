
ι(
6
Add_Prev_Layer(0J 5   KM   K` 
5
LayerNorm_1(0J 5   KM   K` 
7
Q(0:* 5   KE   KM   K` 
7
K(0:* 5   KE   KM   K` 
7
V(0:* 5   KE   KM   K` 
@

MHA_GEMM_1(0B* 5   KE   KM   M`@ 
1
SOFTMAX(0J 5   MM   M`@ 
3
	DropOut_1(0J 5   MM   M`@ 
@

MHA_GEMM_2	(0B* 5   KE   MM   K` 
F
	PROJ_GEMM
(0:1 5   KE   KM   K`hu   K 
3
	DropOut_2(0J 5   KM   K` 
5
Add_1	(0R$ 5   KM   K` υ   K
5
LayerNorm_2
(0J 5   KM   K` 
?
FFN0(0:/@ 5   KE   LM   L`u    
.
GeLU(0J@ 5   LM   L` 
A
FFN1(0:1@ 5   LE   LM   K`hu   K 
3
	DropOut_3(0J 5   KM   K` 
5
Add_2(0R$ 5   KM   K` υ   K
2
Loss_bwd (0J 5   KM   K` 
9
DropOut_3_bwd (0J 5   KM   K` 
@
FFN1_bwd (0:*@ 5   KE   LM   L` 
4
GeLU_bwd (0J@ 5   LM   L` 
G
FFN0_bwd (0:1@ 5   LE   LM   K`hu   K 
;
LayerNorm_2_bwd (0J 5   KM   K` 
9
DropOut_2_bwd (0J 5   KM   K` 
E
PROJ_GEMM_bwd (0:* 5   KE   KM   K` 
G
MHA_GEMM_2_bwd1 (0B* 5   KE   KM   M`@ 
G
MHA_GEMM_2_bwd2 (0B* 5   KE   MM   K` 
D
V_bwd	 (0:1 5   KE   KM   K`hu   K 
9
DropOut_1_bwd	 (0J 5   MM   M`@ 
7
SOFTMAX_bwd
 (0J 5   MM   M`@ 
G
MHA_GEMM_1_bwd1  (0B* 5   ME   KM   K` 
G
MHA_GEMM_1_bwd2! (0B* 5   ME   KM   K` 
=
Q_bwd" (0:* 5   KE   KM   K` 
=
K_bwd# (0:* 5   KE   KM   K` 
U
FFN1_bwd_weight_update$ (0B1@ 5   KE   LM   L`hu   L 
U
FFN0_bwd_weight_update% (0B1 @5   LE   KM   L`hu   L@ 
Z
PROJ_GEMM_bwd_weight_update& (0B1 5   KE   KM   K`hu   K 
R
V_bwd_weight_update'	 (0B1 5   KE   KM   K`hu   K 
R
K_bwd_weight_update( (0B1 5   KE   KM   K`hu   K 
R
Q_bwd_weight_update) (0B1 5   KE   KM   K`hu   K 20=   KE   FPZAdd_Prev_LayerbLayerNorm_1 %0=   KE   FPZLayerNorm_1bQ %0=   KE   FPZLayerNorm_1bK %0=   KE   FPZLayerNorm_1bV $0=   KE   EPZQb
MHA_GEMM_1 $0=   KE   EPZKb
MHA_GEMM_1 *0=   ME   GPZ
MHA_GEMM_1bSOFTMAX )0=   ME   GPZSOFTMAXb	DropOut_1 $	0=   KE   EP	ZVb
MHA_GEMM_2 ,	0=   ME   GP
Z	DropOut_1b
MHA_GEMM_2 ,	
0=   KE   EPZ
MHA_GEMM_2b	PROJ_GEMM +
0=   KE   FPZ	PROJ_GEMMb	DropOut_2 '0=   KE   FPZ	DropOut_2bAdd_1 ,0
=   KE   FPZAdd_Prev_LayerbAdd_1 )0=   KE   FPZAdd_1bLayerNorm_2 (0=   KE   FPZLayerNorm_2bFFN0 !0=   LE   FPZFFN0bGeLU !0=   LE   FPZGeLUbFFN1 &0=   KE   FPZFFN1b	DropOut_3 '0=   KE   EPZ	DropOut_3bAdd_2 #0=   KE   FPZAdd_1bAdd_2 .0=   KE   FPZLoss_bwdbDropOut_3_bwd .0=   KE   FPZDropOut_3_bwdbFFN1_bwd )0=   LE   FPZFFN1_bwdbGeLU_bwd )0=   LE   FPZGeLU_bwdbFFN0_bwd 00=   KE   FPZFFN0_bwdbLayerNorm_2_bwd 50=   KE   FPZLayerNorm_2_bwdbDropOut_2_bwd 30=   KE   FPZDropOut_2_bwdbPROJ_GEMM_bwd 50=   KE   EPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd1 )0=   KE   EPZVbMHA_GEMM_2_bwd1 50=   KE   EPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd2 10=   ME   GP Z	DropOut_1bMHA_GEMM_2_bwd2 50=   ME   GP!ZMHA_GEMM_2_bwd1bDropOut_1_bwd -0=   KE   EP"ZMHA_GEMM_2_bwd2bV_bwd 10=   ME   GP#ZDropOut_1_bwdbSOFTMAX_bwd 3 0=   ME   GP$ZSOFTMAX_bwdbMHA_GEMM_1_bwd1 3!0=   ME   GP%ZSOFTMAX_bwdbMHA_GEMM_1_bwd2 ) 0
=   KE   EP&ZKbMHA_GEMM_1_bwd1 )!0
=   KE   EP'ZQbMHA_GEMM_1_bwd2 - "0=   KE   EP(ZMHA_GEMM_1_bwd1bQ_bwd -!#0=   KE   EP)ZMHA_GEMM_1_bwd2bK_bwd <$0=   KE   FP*ZDropOut_3_bwdbFFN1_bwd_weight_update 3$0=   LE   FP+ZGeLUbFFN1_bwd_weight_update 7%0	=   LE   FP,ZGeLU_bwdbFFN0_bwd_weight_update :%0=   KE   FP-ZLayerNorm_2bFFN0_bwd_weight_update A&0=   KE   FP.ZDropOut_2_bwdbPROJ_GEMM_bwd_weight_update >	&0=   KE   EP/Z
MHA_GEMM_2bPROJ_GEMM_bwd_weight_update ;'0=   KE   EP0ZMHA_GEMM_2_bwd2bV_bwd_weight_update 7'0	=   KE   FP1ZLayerNorm_1bV_bwd_weight_update ;!(0=   KE   EP2ZMHA_GEMM_1_bwd2bK_bwd_weight_update 7(0=   KE   FP3ZLayerNorm_1bK_bwd_weight_update ; )0=   KE   EP4ZMHA_GEMM_1_bwd1bQ_bwd_weight_update 7)0=   KE   FP5ZLayerNorm_1bQ_bwd_weight_update 2° -  ¨LUαz΄?" ­  D²TP’
 `ΒD   Q-   @  ΐA  ?% θ G-Γd*<5τύT=EτύT=M(a&>U ;D"&: (X@`xpxfff?*o:΄