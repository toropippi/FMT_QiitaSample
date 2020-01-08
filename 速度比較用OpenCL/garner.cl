
#define MOD_P0 ((ulong)469762049)
#define MOD_P1 ((ulong)1811939329)
#define MOD_P2 ((ulong)2013265921)


//３つの互いに素なPが与えられるので、それぞれの余りから元の値を復元したい
//このときPは全て固定なので剰余計算は全部決め打ちでいける

//E0～E2が入力、E3に出力
//繰り上がりも考慮
//arrayLength2=arrayE3の配列サイズ
__kernel void Garner(__global uint *arrayE0,__global uint *arrayE1,__global uint *arrayE2,__global uint *arrayE3,uint arrayLength2 ) {
	int idx = get_global_id(0);
	
	ulong ar=arrayE0[idx];
	ulong br=arrayE1[idx];
	ulong cr=arrayE2[idx];
	
	ulong x=ar;
	ulong brx=br-x+MOD_P1;
	if (brx>=MOD_P1)brx-=MOD_P1;
	x=x+(brx*(ulong)1540148431)%MOD_P1*MOD_P0;
	//1540148431=modinv(MOD_P0,MOD_P1)
	//この時点でxはMOD_P1*MOD_P0以下であることが保証されている

	ulong crx=cr+MOD_P2-x%MOD_P2;
	if (crx>=MOD_P2)crx-=MOD_P2;
	ulong w1=(crx*(ulong)1050399624)%MOD_P2;
	//1050399624=modinv(MOD_P0,MOD_P2) *modinv(MOD_P1,MOD_P2)%MOD_P2
	ulong w2=MOD_P0*MOD_P1;
	ulong out_lo=w1*w2;
	ulong out_hi=mul_hi(w1,w2);
	
	if (out_lo>(out_lo+x)){
		out_hi++;
	}
	out_lo+=x;
	
	//ここから繰り上がり処理
	uint ui00_32=(uint)(out_lo%((ulong)4294967296));
	uint ui32_64=(uint)(out_lo/((ulong)4294967296));
	uint ui64_96=(uint)(out_hi%((ulong)4294967296));
	
	uint lastE3_0 = atomic_add( &arrayE3[idx+0], ui00_32 );
	if ((lastE3_0+ui00_32)<lastE3_0){//繰り上がりを考慮
		ui32_64++;
		if (ui32_64==0)ui64_96++;
	}
	
	if ((ui32_64!=0)&(idx<arrayLength2-1)){
		uint lastE3_1 = atomic_add( &arrayE3[idx+1], ui32_64 );
		if ((lastE3_1+ui32_64)<lastE3_1){//繰り上がりを考慮
			ui64_96++;//こいつがオーバーフローすることは絶対にない
		}
	}
	
	uint upflg=0;
	if ((ui64_96!=0)&(idx<arrayLength2-2)){
		uint lastE3_2 = atomic_add( &arrayE3[idx+2], ui64_96 );
		if ((lastE3_2+ui64_96)<lastE3_2){//繰り上がりを考慮
			upflg++;
		}
	}
	
	uint lastE3_i;
	for(int i=idx+3;i<arrayLength2;i++){ //9999999+1みたいなとき用
		if (upflg==0)break;
		lastE3_i = atomic_add( &arrayE3[i], upflg );
		if (lastE3_i==4294967295){
			upflg=1;
		}else{
			upflg=0;
		}
	}
	
}