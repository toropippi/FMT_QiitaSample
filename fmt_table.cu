#define ulong unsigned long long
#define uint unsigned int


//A*B%MODP
__device__ uint ABModC(uint a,uint b){
	ulong tmp=((ulong)(__umulhi(a,b)))*(1ULL<<32)+(ulong)(a*b);
	return (uint)(tmp%MODP);
}

//exp(a,b)%MODP
__device__ uint ModExp(uint a,uint b){
	ulong ans=1ULL;
	ulong aa=a;
	
	while(b!=0){
		if (b%2==1) ans=ans*aa%MODP;
		aa=aa*aa%MODP;
		b/=2;
	}
	return (uint)ans;
}


//逆変換後は、FFTでいうNで除算しないといけない。
// a/arrayLength mod P
__device__ uint DivN_f(uint a,uint arrayLength)
{
	uint as    =a/arrayLength;
	uint ar    =a-as*arrayLength;
	uint pn    =MODP/arrayLength;
	if (ar!=0){
		as+=(arrayLength-ar)*pn+1;
	}
	return as;
}




__global__ void uFMT(uint *arrayA,uint loopCnt,uint omega,uint arrayLength ,uint *mtable) {
	uint idx = threadIdx.x+blockIdx.x*256;
	uint loopCnt_Pow2=1<<loopCnt;
	uint t2 = idx%loopCnt_Pow2;
	uint t0 = idx*2-t2;
	uint t1 = t0+loopCnt_Pow2;
	uint w0;
	uint w1;
	uint arrayAt0=arrayA[t0];
	uint arrayAt1=arrayA[t1];
	uint r0;
	uint r1;
	uint ridx=t2*(arrayLength>>(loopCnt+1));
	if (ridx>=arrayLength)ridx-=arrayLength;
	w0=mtable[ridx];
	//w0=ModExp(omega,t2*(arrayLength2/loopCnt_Pow2));
	r0=arrayAt0-arrayAt1+MODP;
	r1=arrayAt0+arrayAt1;
	if (r0>=MODP){r0-=MODP;}
	if (r1>=MODP){r1-=MODP;}
	w1=ABModC(r0,w0);
	arrayA[t1]=w1;
	arrayA[t0]=r1;	
}


__global__ void iFMT(uint *arrayA,uint loopCnt,uint omega,uint arrayLength ,uint *mtable) {
	uint idx = threadIdx.x+blockIdx.x*256;
	uint loopCnt_Pow2=1<<loopCnt;
	uint t2 = idx%loopCnt_Pow2;
	uint t0 = idx*2-t2;
	uint t1 = t0+loopCnt_Pow2;
	uint w0;
	uint w1;
	uint arrayAt0=arrayA[t0];
	uint arrayAt1=arrayA[t1];
	uint r0;
	uint r1;
	uint ridx=arrayLength-t2*(arrayLength>>(loopCnt+1));
	if (ridx>=arrayLength)ridx-=arrayLength;
	w0=mtable[ridx];
	//w0=ModExp(omega,arrayLength2*2-t2*(arrayLength2/loopCnt_Pow2));
	w1=ABModC(arrayAt1,w0);
	r0=arrayAt0-w1+MODP;
	r1=arrayAt0+w1;
	if (r0>=MODP){r0-=MODP;}
	if (r1>=MODP){r1-=MODP;}
	arrayA[t1]=r0;
	arrayA[t0]=r1;
}


//同じ要素同士の掛け算
__global__ void Mul_i_i(uint *arrayA,uint *arrayB ) {
	uint idx = threadIdx.x+blockIdx.x*256;
	uint w0;
	w0=ABModC(arrayB[idx],arrayA[idx]);
	arrayB[idx]=w0;
}

//逆変換後のNで割るやつ。剰余下で割るには特殊処理が必要
__global__ void DivN(uint *arrayA,uint arrayLength ) {
	uint idx = threadIdx.x+blockIdx.x*256;
	arrayA[idx]=DivN_f(arrayA[idx],arrayLength);
}



//負巡回計算の前処理
//sqrt_omegaの2N乗が1 (mod P)
//a[0]*=ModExp(sqrt_omega,0)
//a[1]*=ModExp(sqrt_omega,1)
//a[2]*=ModExp(sqrt_omega,2)
//a[3]*=ModExp(sqrt_omega,3)
__global__ void PreNegFMT(uint *arrayA,uint *arrayB,uint sqrt_omega,uint *mtable,uint arrayLength) {
	uint idx = threadIdx.x+blockIdx.x*256;
	//w0=ModExp(sqrt_omega,idx);
	uint w0=mtable[idx/2];
	if (idx%2==1)
		w0=ABModC(sqrt_omega,w0);
	arrayA[idx]%=MODP;//これは本来必要ないが、一番最初に入力されたA,Bが必ずMODPの剰余下の値になっているとは限らないので
	arrayB[idx]=ABModC(arrayA[idx],w0);
}

//負巡回計算の後処理
//sqrt_omegaの2N乗が1 (mod P)
//a[0]*=ModExp(sqrt_omega,-0)
//a[1]*=ModExp(sqrt_omega,-1)
//a[2]*=ModExp(sqrt_omega,-2)
//a[3]*=ModExp(sqrt_omega,-3)
__global__ void PostNegFMT(uint *arrayA,uint sqrt_omega,uint *mtable,uint arrayLength) {
	uint idx = threadIdx.x+blockIdx.x*256;
	//uint w0=ModExp(sqrt_omega,arrayLength*2-idx);	
	uint w0=mtable[(arrayLength*2-idx)%(arrayLength*2)/2];
	if (idx%2==1)
		w0=ABModC(sqrt_omega,w0);
	arrayA[idx]=ABModC(arrayA[idx],w0);
}

//負巡回計算と正巡回計算結果から、上半分桁と下半分桁を求める
__global__ void PosNeg_To_HiLo(uint *arrayE,uint *arrayA,uint *arrayB,uint arrayLength) {
	uint idx = threadIdx.x+blockIdx.x*256;
	uint a=arrayA[idx];
	uint b=arrayB[idx];
	uint subab=(a-b+MODP);//まず(a-b)/2を求めたい
	uint flag=subab%2;
	subab-=MODP*((subab>=MODP)*2-1)*flag;//ここで絶対偶数になる
	subab/=2;//(a-b)/2 MOD Pを算出
	arrayE[idx+arrayLength]=subab;//上位桁は(a-b)/2 MOD P
	arrayE[idx]=a-subab+MODP*(a<subab);//a-((a-b)/2)=a/2+b/2 つまり(a+b)/2が下位桁
}


//vramへの書き込み回数を減らす目的に作った関数
//PostNegFMT関数とDivN関数とPosNeg_To_HiLo関数の統合版
__global__ void PostFMT_DivN_HiLo(uint *arrayE,uint *arrayA,uint *arrayB,uint arrayLength,uint sqrt_omega) {
	uint idx = threadIdx.x+blockIdx.x*256;
	uint a=arrayA[idx];
	uint b=arrayB[idx];

	//ここは負巡回の後処理計算部分
	uint w0=ModExp(sqrt_omega,idx+(idx%2)*arrayLength);
	b=ABModC(b,w0);
	
	//Nで除算する関数
	a=DivN_f(a,arrayLength);
	b=DivN_f(b,arrayLength);
	
	//あとは一緒
	uint subab=(a-b+MODP);//まず(a-b)/2を求めたい
	uint flag=subab%2;
	subab-=MODP*((subab>=MODP)*2-1)*flag;//ここで絶対偶数になる
	subab/=2;//(a-b)/2 MOD Pを算出
	arrayE[idx+arrayLength]=subab;//上位桁は(a-b)/2 MOD P
	arrayE[idx]=a-subab+MODP*(a<subab);//a-((a-b)/2)=a/2+b/2 つまり(a+b)/2が下位桁
}



//最初にべき乗余を計算する関数
__global__ void CreateTable(uint *mtable,uint omega) {
	uint idx = threadIdx.x+blockIdx.x*256;
	mtable[idx]=ModExp(omega,idx);
}