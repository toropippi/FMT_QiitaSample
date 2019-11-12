#define ulong unsigned long long
#define uint unsigned int



// B=A%MODP
__global__ void CreateAmod(uint *arrayB,uint *arrayA,uint modp) {
	uint idx = threadIdx.x+blockIdx.x*256;
	arrayB[idx]=arrayA[idx]%modp;
}





//C=A-B
__global__ void SubCAB(uint *arrayC,uint *arrayA,uint *arrayB,uint arrayLength) {
	int idx = threadIdx.x+blockIdx.x*256;
	uint flag=0;
	if (idx==0)flag=1;
	
	uint b=4294967295-arrayB[idx];
	uint a=arrayA[idx];
	
	uint upflag=0;
	
	if ((a==4294967295)&(flag==1)){
		upflag=1;
		a=0;
	}
	
	uint c=a+b;
	if (c<b){
		upflag=1;
	}
	
	
	uint lastC_0 = atomicAdd( &arrayC[idx+0], c );
	
	if ((lastC_0+c)<lastC_0){//繰り上がりを考慮
		upflag++;
	}
	
	uint lastC_i;
	for(int i=idx+1;i<arrayLength;i++){ //9999999+1みたいなとき用
		if (upflag==0)break;
		lastC_i = atomicAdd( &arrayC[i], upflag );
		if ((lastC_i+upflag)<lastC_i){
			upflag=1;
		}else{
			upflag=0;
		}
	}
}






//C=A+B
__global__ void AddCAB(uint *arrayC,uint *arrayA,uint *arrayB,uint arrayLength) {
	int idx = threadIdx.x+blockIdx.x*256;
	
	uint b=arrayB[idx];
	uint a=arrayA[idx];
	
	uint upflag=0;
	
	uint c=a+b;
	if (c<b){
		upflag=1;
	}
	
	
	uint lastC_0 = atomicAdd( &arrayC[idx+0], c );
	
	if ((lastC_0+c)<lastC_0){//繰り上がりを考慮
		upflag++;
	}
	
	uint lastC_i;
	for(int i=idx+1;i<arrayLength;i++){ //9999999+1みたいなとき用
		if (upflag==0)break;
		lastC_i = atomicAdd( &arrayC[i], upflag );
		if ((lastC_i+upflag)<lastC_i){
			upflag=1;
		}else{
			upflag=0;
		}
	}
	
}








//C=0
__global__ void SetZero(uint *arrayC) {
	int idx = threadIdx.x+blockIdx.x*256;
	arrayC[idx]=0;
}




//Bは最初0に初期化必要
//B=A*(1<<n)
__global__ void ShiftBA(uint *arrayB,uint *arrayA,int n,uint arrayLength) {
	int idx = threadIdx.x+blockIdx.x*256;
	
	uint a=arrayA[idx];
	
	int nn=n/32;
	n-=nn*32;
	if (n<0){
		n+=32;
		nn-=1;
	}
	uint sa0=a<<((uint)n);//下の位
	uint sa1=a>>((uint)(32-n));//下の位
	
	if ((idx+nn<arrayLength)&((idx+nn>=0))){
		atomicAdd( &arrayB[idx+nn], sa0 );
	}
	if (((idx+nn+1)<arrayLength)&(((idx+nn+1)>=0))){
		atomicAdd( &arrayB[idx+nn+1], sa1 );
	}

}




//memcpy_uint ver
//allow overflow
__global__ void memcpy_dtod_ui4(uint *dst,uint *src,uint dstoffset,uint srcoffset,uint copysize) {
	int idx = threadIdx.x+blockIdx.x*256;
	if (idx<copysize){
		dst[idx+dstoffset]=src[idx+srcoffset];
	}
}



//ニュートン法でやる最初の逆数の計算
//outに2要素で出力
__global__ void FirstNewtonRev(uint *out,uint *A,uint index) {
	ulong a=A[index];
	ulong b=18446744073709551615;
	ulong c=b/a;
	if ((b%a)==(a-1)){
		c++;
	}
	uint c0=c%4294967296;
	uint c1=c/4294967296;
	out[0]=c0;
	out[1]=c1;
}



//ニュートン法でやる最初の逆数の計算
//outに2要素で出力
__global__ void FirstNewtonSqrt(uint *out,uint *A,uint index) {
	double a=(double)A[index];
	a/=4294967296.0;
	a=1.0/sqrt(a);
	a*=4294967296.0;
	ulong c=(ulong)a;
	uint c0=c%4294967296;
	uint c1=c/4294967296;
	out[0]=c0;
	out[1]=c1;
}