#define ulong unsigned long long
#define uint unsigned int

#define MOD_P0 469762049LL
#define MOD_P1 1811939329LL
#define MOD_P2 2013265921LL


//３つの互いに素なPが与えられるので、それぞれの余りから元の値を復元したい
//このときPは全て固定なので剰余計算は全部決め打ちでいける

//E0〜E2が入力、E3に出力
//繰り上がりも考慮
//arrayLength2=arrayE3の配列サイズ
__global__ void GarnerGPU(uint *arrayE0,uint *arrayE1,uint *arrayE2,uint *arrayE3,uint arrayLength2 ) {
	int idx = threadIdx.x+blockIdx.x*256;
	
	ulong ar=arrayE0[idx];
	ulong br=arrayE1[idx];
	ulong cr=arrayE2[idx];
	
	ulong x=ar;
	ulong brx=br-x+MOD_P1;
	if (brx>=MOD_P1)brx-=MOD_P1;
	x=x+(brx*1540148431)%MOD_P1*MOD_P0;
	//1540148431=modinv(MOD_P0,MOD_P1)
	//この時点でxはMOD_P1*MOD_P0以下であることが保証されている

	ulong crx=cr+MOD_P2-x%MOD_P2;
	if (crx>=MOD_P2)crx-=MOD_P2;
	ulong w1=(crx*1050399624)%MOD_P2;
	//1050399624=modinv(MOD_P0,MOD_P2) *modinv(MOD_P1,MOD_P2)%MOD_P2
	ulong w2=MOD_P0*MOD_P1;
	ulong out_lo=w1*w2;
	ulong out_hi=__umul64hi(w1,w2);
	
	if (out_lo>(out_lo+x)){
		out_hi++;
	}
	out_lo+=x;
	
	//ここから繰り上がり処理
	uint ui00_32=(uint)(out_lo%(1ULL<<32ULL));
	uint ui32_64=(uint)(out_lo/(1ULL<<32ULL));
	uint ui64_96=(uint)(out_hi%(1ULL<<32ULL));
	
	uint lastE3_0 = atomicAdd( &arrayE3[idx+0], ui00_32 );
	if ((lastE3_0+ui00_32)<lastE3_0){//繰り上がりを考慮
		ui32_64++;
		if (ui32_64==0)ui64_96++;
	}
	
	if (ui32_64!=0){
		uint lastE3_1 = atomicAdd( &arrayE3[idx+1], ui32_64 );
		if ((lastE3_1+ui32_64)<lastE3_1){//繰り上がりを考慮
			ui64_96++;//こいつがオーバーフローすることは絶対にない
		}
	}
	
	uint upflg=0;
	if (ui64_96!=0){
		uint lastE3_2 = atomicAdd( &arrayE3[idx+2], ui64_96 );
		if ((lastE3_2+ui64_96)<lastE3_2){//繰り上がりを考慮
			upflg++;
		}
	}
	
	uint lastE3_i;
	for(int i=idx+3;i<arrayLength2;i++){ //9999999+1みたいなとき用
		if (upflg==0)break;
		lastE3_i = atomicAdd( &arrayE3[i], upflg );
		if (lastE3_i==4294967295){
			upflg=1;
		}else{
			upflg=0;
		}
	}
}