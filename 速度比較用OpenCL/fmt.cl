

//A*B%MODP
uint ABModC(uint a,uint b){
	ulong tmp=((ulong)(mul_hi(a,b)))*((ulong)4294967296)+(ulong)(a*b);
	return (uint)(tmp%(ulong)MODP);
}

//exp(a,b)%MODP
uint ModExp(uint a,uint b){
	ulong ans=1;
	ulong aa=a;
	
	while(b!=0){
		if (b%2==1) ans=ans*aa%(ulong)MODP;
		aa=aa*aa%(ulong)MODP;
		b/=2;
	}
	return (uint)ans;
}


// a/arrayLength mod P
uint DivN_f(uint a,uint arrayLength)
{
	uint as    =a/arrayLength;
	uint ar    =a-as*arrayLength;
	uint pn    =MODP/arrayLength;
	if (ar!=0){
		as+=(arrayLength-ar)*pn+1;
	}
	return as;
}






__kernel void uFMT(__global uint *arrayA,uint loopCnt_Pow2,uint omega,uint arrayLength2 ) {
	uint idx = get_global_id(0);
	uint t2 = idx%loopCnt_Pow2;
	uint t0 = idx*2-t2;
	uint t1 = t0+loopCnt_Pow2;
	uint w0;
	uint w1;
	uint arrayAt0=arrayA[t0];
	uint arrayAt1=arrayA[t1];
	uint r0;
	uint r1;
	w0=ModExp(omega,t2*(arrayLength2/loopCnt_Pow2));
	r0=arrayAt0-arrayAt1+MODP;
	r1=arrayAt0+arrayAt1;
	if (r0>=MODP){r0-=MODP;}
	if (r1>=MODP){r1-=MODP;}
	w1=ABModC(r0,w0);
	arrayA[t1]=w1;
	arrayA[t0]=r1;
}


__kernel void iFMT(__global uint *arrayA,uint loopCnt_Pow2,uint omega,uint arrayLength2 ) {
	uint idx = get_global_id(0);
	uint t2 = idx%loopCnt_Pow2;
	uint t0 = idx*2-t2;
	uint t1 = t0+loopCnt_Pow2;
	uint w0;
	uint w1;
	uint arrayAt0=arrayA[t0];
	uint arrayAt1=arrayA[t1];
	uint r0;
	uint r1;
	w0=ModExp(omega,arrayLength2*2-t2*(arrayLength2/loopCnt_Pow2));
	w1=ABModC(arrayAt1,w0);
	r0=arrayAt0-w1+MODP;
	r1=arrayAt0+w1;
	if (r0>=MODP){r0-=MODP;}
	if (r1>=MODP){r1-=MODP;}
	arrayA[t1]=r0;
	arrayA[t0]=r1;
}


__kernel void Mul_i_i(__global uint *arrayA,__global uint *arrayB ) {
	uint idx = get_global_id(0);
	uint w0;
	w0=ABModC(arrayB[idx],arrayA[idx]);
	arrayB[idx]=w0;
}


__kernel void DivN(__global uint *arrayA,uint arrayLength ) {
	uint idx = get_global_id(0);
	arrayA[idx]=DivN_f(arrayA[idx],arrayLength);
}





//a[0]*=ModExp(sqrt_omega,0)
//a[1]*=ModExp(sqrt_omega,1)
//a[2]*=ModExp(sqrt_omega,2)
//a[3]*=ModExp(sqrt_omega,3)
__kernel void PreNegFMT(__global uint *arrayA,__global uint *arrayB,uint sqrt_omega,uint arrayLength) {
	uint idx = get_global_id(0);
	uint ara=arrayA[idx]%MODP;//本来必要ないが入力A,Bはかならず剰余されてないと正しく計算できないのでこの関数にまとめてしまう
	arrayA[idx]=ara;
	uint w0=ModExp(sqrt_omega,idx);
	arrayB[idx]=ABModC(ara,w0);
}


//a[0]*=ModExp(sqrt_omega,-0)
//a[1]*=ModExp(sqrt_omega,-1)
//a[2]*=ModExp(sqrt_omega,-2)
//a[3]*=ModExp(sqrt_omega,-3)
__kernel void PostNegFMT(__global uint *arrayA,uint sqrt_omega,uint arrayLength) {
	uint idx = get_global_id(0);
	uint w0=ModExp(sqrt_omega,arrayLength*2-idx);
	arrayA[idx]=ABModC(arrayA[idx],w0);
}


__kernel void PosNeg_To_HiLo(__global uint *arrayE,__global uint *arrayA,__global uint *arrayB,uint arrayLength) {
	uint idx = get_global_id(0);
	uint a=arrayA[idx];
	uint b=arrayB[idx];
	uint subab=(a-b+MODP);
	uint flag=subab%2;
	subab-=MODP*((subab>=MODP)*2-1)*flag;
	subab/=2;
	arrayE[idx+arrayLength]=subab;
	arrayE[idx]=a-subab+MODP*(a<subab);
}

