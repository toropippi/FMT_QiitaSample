#include "bits/stdc++.h"
#include <time.h>
#include <random>
#include <fstream>

using namespace std;
#define ll long long
#define ull unsigned long long
#define uint unsigned int


random_device rd;
mt19937 mt(rd()+(uint)time(NULL));

uint digit_level;
uint digitN;

uint MODP;
uint MODP_WnSqrt;
uint MODP_Wn;
const uint MODP0=469762049;
const uint MODP1=1811939329;
const uint MODP2=2013265921;

vector<uint> MTABLE0;
vector<uint> MTABLE1;
vector<uint> MTABLE2;
vector<uint> *pMTABLE;


uint CreateWnSqrt(uint a,uint m,uint n_level){
  ull ans=a;
  ull ulm=m;
  for(int i=0;i<n_level-digit_level-1;i++){
    ans=ans*ans%ulm;
  }
  return (uint)ans;
}

//aの逆元を返す mod m
uint modinv(uint a,uint m){
    ll b = m;
    ll u = 1;
    ll v = 0;
    ll t,tmp;
    while (b!=0){
        t = a / b;
        a -= t * b;
        tmp=a;
        a=b;
        b=tmp;
        u -= t * v;
        tmp = u;
        u = v;
        v = tmp;
    }
    u %= m;
    if (u < 0)u += m;
    return (uint)u;
}

inline uint expmod(uint n){
  if (n>=digitN)n-=digitN;
  return (*pMTABLE)[n];
}


//garnerのアルゴリズム、3変数ver
void Garner(ll a_,ll b_,ll c_,ll ar_,ll br_,ll cr_,uint *rx_lo,uint *rx_md,uint *rx_hi){
  ll a = a_;
  ll b = b_;
  ll c = c_;
  ll ar = ar_;
  ll br = br_;
  ll cr = cr_;
  ll mab=(ll)modinv(a,b);
  ll x=(br - ar) * mab % b;
  if (x<0){
    x+=b;
  }
  x = ar + x * a;

  ll tmp=((cr - x % c)+c)%c;
  ull px=tmp * (ull)modinv(a,c) % c * (ull)modinv(b,c) % c * a;//*b

  ull px_hi=px/(1ULL<<32ULL);
  ull px_lo=px%(1ULL<<32ULL);
  px_hi*=b;
  px_lo*=b;
  px_hi+=px_lo/(1ULL<<32ULL);
  px_lo%=(1ULL<<32ULL);
  px_lo+=x;
  px_hi+=px_lo/(1ULL<<32ULL);
  px_lo%=(1ULL<<32ULL);

  *rx_lo=(uint)(px_lo);
  *rx_md=(uint)(px_hi%(1ULL<<32ULL));
  *rx_hi=(uint)(px_hi/(1ULL<<32ULL));
  }


void iFMT(vector<uint>& arrayA){
  uint t2,t0,t1,arrayAt0,arrayAt1,w0,w1;
    for(int i=0;i<digit_level;i++){
      uint loopCnt_Pow2 = 1 << i;
      for(int idx=0;idx<digitN/2;idx++){
        t2 = idx % loopCnt_Pow2;
        t0 = idx * 2 - t2;
        t1 = t0 + loopCnt_Pow2;
        arrayAt0 = arrayA[t0];
        arrayAt1 = arrayA[t1];
        w0 = expmod(digitN - t2 * (1 << (digit_level - 1 - i)));
        w1 = (uint)((ull)arrayAt1 * (ull)w0 % (ull)MODP);
        arrayAt1=arrayAt0 + MODP - w1;
        if (arrayAt1>=MODP)arrayAt1-=MODP;
        arrayAt0+=w1;
        if (arrayAt0>=MODP)arrayAt0-=MODP;

        arrayA[t1] = arrayAt1;
        arrayA[t0] = arrayAt0;
      }
    }
  }


void uFMT(vector<uint>& arrayA){
  uint t2,t0,t1,arrayAt0,arrayAt1,w0,w1,r0,r1;
    for(int i=0;i<digit_level;i++){
      uint loopCnt_Pow2 = 1 << (digit_level - i - 1);
      for(int idx=0;idx<digitN/2;idx++){
          t2 = idx % loopCnt_Pow2;
          t0 = idx * 2 - t2;
          t1 = t0 + loopCnt_Pow2;
          arrayAt0 = arrayA[t0];
          arrayAt1 = arrayA[t1];
          w0 = expmod(t2 * (1 << i));
          r0 = (arrayAt0 - arrayAt1 + MODP);
          if (r0>=MODP)r0-= MODP;
          r1 = (arrayAt0 + arrayAt1);
          if (r1>=MODP)r1-= MODP;
          w1 = (uint)((ull)r0 * (ull)w0 % (ull)MODP);
          arrayA[t1] = w1;
          arrayA[t0] = r1;
      }
    }
  }


void DivN(vector<uint>& arrayA){
  ull digitN_r = modinv(digitN, MODP);
  for(int i=0;i<digitN;i++){
    ull ret = (ull)(arrayA[i]) * digitN_r % (ull)MODP;
    arrayA[i]=(uint)(ret);
  }
}


void PreNegFMT(vector<uint>& arrayA){
  ull mws=(ull)MODP_WnSqrt;
  ull expmodi=1;
  for(int i=0;i<digitN;i++){
    arrayA[i] = (uint)((ull)arrayA[i] * expmodi % (ull)MODP);
    expmodi=expmodi*mws%(ull)MODP;
  }
}

void PostNegFMT(vector<uint>& arrayA){
  ull mws=(ull)modinv(MODP_WnSqrt,MODP);
  ull expmodi=1;
  for(int i=0;i<digitN;i++){
    arrayA[i] = (uint)((ull)arrayA[i] * expmodi % (ull)MODP);
    expmodi=expmodi*mws%(ull)MODP;
  }
}


//    #A*Bの畳み込み結果を返す
//    #上位桁、下位桁を復元して2倍の要素数で返す
void Convolution(vector<uint>& A_,vector<uint>& B_,vector<uint>& E,int pno){
  //#正巡回
  vector<uint> A1(digitN),B1(digitN),A2(digitN),B2(digitN),Cpos(digitN),Cneg(digitN);
  for(int i=0;i<digitN;i++){
    A2[i]=A1[i]=A_[i]%MODP;
  }
  for(int i=0;i<digitN;i++){
    B2[i]=B1[i]=B_[i]%MODP;
  }
  uFMT(A1);
  uFMT(B1);
  for(int i=0;i<digitN;i++){
    Cpos[i]=(uint)((ull)A1[i]*(ull)B1[i]%(ull)MODP);
  }
  iFMT(Cpos);
  DivN(Cpos);


  //# 負巡回
  PreNegFMT(A2);
  PreNegFMT(B2);
  uFMT(A2);
  uFMT(B2);
  for(int i=0;i<digitN;i++){
    Cneg[i]=(uint)((ull)A2[i]*(ull)B2[i]%(ull)MODP);
  }
  iFMT(Cneg);
  DivN(Cneg);
  PostNegFMT(Cneg);

  for(int i=0;i<digitN;i++){
    //#上位桁、下位桁 復元
    uint Elo = (Cpos[i] + Cneg[i]) % MODP;
    Elo = (Elo + Elo % 2 * MODP) /2;

    uint Ehi = (Cpos[i] - Cneg[i] + MODP) % MODP;
    Ehi = (Ehi + Ehi % 2 * MODP) / 2;

    E[i] = Elo;
    E[digitN+i] = Ehi;
  }
}

//#E0,E1,E2を使ってgarnerでEを復元して繰り上がり処理
void Carrying(vector<uint>& E0,vector<uint>& E1,vector<uint>& E2,vector<uint>& E){
  uint upg=0;
  uint out_lo,out_md,out_hi;
  for(int i=0;i<digitN*2;i++){
    upg=0;
    Garner(MODP0,MODP1,MODP2, E0[i], E1[i], E2[i],&out_lo,&out_md,&out_hi);
    
    if (E[i]>E[i]+out_lo){
      upg=1;
    }
    E[i]+=out_lo;

    if (out_md==4294967295){
      if (upg==1)out_hi++;
    }
    out_md+=upg;
    if (i<digitN*2-1){
      if (E[i+1]>E[i+1]+out_md){
        out_hi++;
      }
    }
    E[i+1]+=out_md;

    upg=0;
    if (i<digitN*2-2){
      if (E[i+2]>E[i+2]+out_hi){
        upg=1;
      }
    }
    E[i+2]+=out_hi;
    
    for(int j=i+3;j<digitN*2;j++){
      if (upg==0)break;
      E[j]+=upg;
      if (E[j]!=0)upg=0;
    }

  }
}


void SetMod(int pno)
{
  if (pno==0)
  {
    MODP=MODP0;
    MODP_WnSqrt=CreateWnSqrt(60733,469762049,26);
    MODP_Wn=(uint)((ull)MODP_WnSqrt*(ull)MODP_WnSqrt%(ull)MODP);
    pMTABLE=&MTABLE0;
  }
  if (pno==1)
  {
    MODP=MODP1;
    MODP_WnSqrt=CreateWnSqrt(59189,1811939329,26);
    MODP_Wn=(uint)((ull)MODP_WnSqrt*(ull)MODP_WnSqrt%(ull)MODP);
    pMTABLE=&MTABLE1;
  }
  if (pno==2)
  {
    MODP=MODP2;
    MODP_WnSqrt=CreateWnSqrt(52278,2013265921,27);
    MODP_Wn=(uint)((ull)MODP_WnSqrt*(ull)MODP_WnSqrt%(ull)MODP);
    pMTABLE=&MTABLE2;
  }
}


void CreateTable()
{
  MTABLE0.resize(digitN);
  MTABLE1.resize(digitN);
  MTABLE2.resize(digitN);
  MTABLE0[0]=1;
  MTABLE1[0]=1;
  MTABLE2[0]=1;
  SetMod(0);
  for(int i=1;i<digitN;i++){
    MTABLE0[i]=(uint)((ull)MTABLE0[i-1]*(ull)MODP_Wn%(ull)MODP);
  }
  SetMod(1);
  for(int i=1;i<digitN;i++){
    MTABLE1[i]=(uint)((ull)MTABLE1[i-1]*(ull)MODP_Wn%(ull)MODP);
  }
  SetMod(2);
  for(int i=1;i<digitN;i++){
    MTABLE2[i]=(uint)((ull)MTABLE2[i-1]*(ull)MODP_Wn%(ull)MODP);
  }
}

/*
void fSave(vector<uint> &svdata,int cnt)
{
  //ファイル名
  string filename=to_string(cnt);
	// バイナリ出力モードで開く
	fstream file(filename.c_str(), ios::binary | ios::out);
	// 書き込む
	file.write((char*)&svdata[0], svdata.size()*4);
	// 閉じる
	file.close();
}
*/


int main(){
  cout<<"please enter the digit_level 2-25"<<endl;
  cin>>digit_level;
  cout<<"start!!...."<<endl;
  
  digitN=1<<digit_level;//乗算前の要素数。乗算結果は2*digitN要素数になる
  CreateTable();

  vector<uint> A_(digitN),B_(digitN),E0(digitN*2),E1(digitN*2),E2(digitN*2);
  vector<uint> E(digitN*2,0);//結果格納変数
  for(int i=0;i<digitN;i++){
    A_[i]=mt();//ランダム初期値生成
    B_[i]=mt();//ランダム初期値生成
  }

  uint starttime=clock();
  //Pだけ変えただけで、同じのを3つべた書きしている
  SetMod(0);
  Convolution(A_,B_,E0,0);
 
  SetMod(1);
  Convolution(A_,B_,E1,1);

  SetMod(2);
  Convolution(A_,B_,E2,2);
  
  Carrying(E0,E1,E2,E);
  cout<<"calc_time\t\t"<<(clock()-starttime)<<"msec"<<endl;
  //結果出力
  /*
  fSave(A_,0);
  fSave(B_,1);
  fSave(E,2);
  */
  return 0;
}