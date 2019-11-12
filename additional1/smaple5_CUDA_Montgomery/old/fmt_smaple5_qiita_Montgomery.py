# FMT(高速剰余変換) PyCUDA版
# 3つの素数PのもとでFMTを行うサンプル
# ボトルネックだった剰余をモンゴメリ乗算にかえたver
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import time
import os

digit_level=25#################################可変パラメーター############################# 最大25
digitN=1<<digit_level

gsz = 1 << (digit_level - 1) # gpu global_work_size
lsz = min(gsz,256) # gpu local_work_size

gsz2 = digitN
lsz2 = min(gsz2,256)

programid_garner=SourceModule("""
    #include "garner.cu"
""",include_dirs=[os.getcwd()])
kernel_garner=programid_garner.get_function("GarnerGPU")


class FMTClass():
    def __init__(self,MODP,MODP_WnSqrt):
        self.MODP=np.uint32(MODP)
        self.MODP_WnSqrt = np.uint32(MODP_WnSqrt)
        self.MODP_Wn = np.uint32(np.uint64(self.MODP_WnSqrt) * np.uint64(self.MODP_WnSqrt) % np.uint64(self.MODP))
        NB, R2, MASK, NR=self.CreateMontgomery(int(MODP))
        self.programid = SourceModule("""
            #define MODP (""" + str(MODP) + """)
            #define NB (""" + str(NB) + """)
            #define R2 (""" + str(R2) + """)
            #define MASK (""" + str(MASK) + """)
            #define NR (""" + str(NR) + """)
                #include "fmt.cu"
            """, include_dirs=[os.getcwd()])
        self.kernel_FMT=self.programid.get_function("FMT")
        self.kernel_iFMT=self.programid.get_function("iFMT")
        self.kernel_uFMT=self.programid.get_function("uFMT")
        self.kernel_iuFMT=self.programid.get_function("iuFMT")
        self.kernel_Mul_i_i=self.programid.get_function("Mul_i_i")
        self.kernel_PostNegFMT=self.programid.get_function("PostNegFMT")
        self.kernel_PreNegFMT=self.programid.get_function("PreNegFMT")
        self.kernel_DivN=self.programid.get_function("DivN")
        self.kernel_PosNeg_To_HiLo=self.programid.get_function("PosNeg_To_HiLo")
        self.kernel_PostFMT_DivN_HiLo=self.programid.get_function("PostFMT_DivN_HiLo")

    def CreateMontgomery(self,m):
        NB = m.bit_length()
        R2 = (1 << (NB * 2)) % m
        MASK = (1 << NB) - 1
        NR = 0
        t = 0
        vi = 1
        for _ in range(NB):
            if t & 1 == 0:
                t += m
                NR += vi
            t >>= 1
            vi <<= 1
        return NB,R2,MASK,NR

    def FMT(self,gpuMemA):
        for i in range(digit_level):
            self.kernel_FMT(gpuMemA, np.uint32(1 << i), self.MODP_Wn, np.uint32(1 << (digit_level - 1)),
                         grid=(gsz // lsz, 1, 1), block=(lsz, 1, 1))
        return

    def uFMT(self,gpuMemA):
        for i in range(digit_level):
            self.kernel_uFMT(gpuMemA, np.uint32((1<<(digit_level-1))>>i), self.MODP_Wn, np.uint32(1 << (digit_level - 1)),
                         grid=(gsz // lsz, 1, 1), block=(lsz, 1, 1))
        return

    def iFMT(self,gpuMemA):
        for i in range(digit_level):
            self.kernel_iFMT(gpuMemA, np.uint32(1 << i), self.MODP_Wn,
                         np.uint32(1 << (digit_level - 1)), grid=(gsz // lsz, 1, 1), block=(lsz, 1, 1))
        return

    def iuFMT(self,gpuMemA):
        for i in range(digit_level):
            self.kernel_iuFMT(gpuMemA, np.uint32(1<<(digit_level-1-i)), self.MODP_Wn,
                         np.uint32(1 << (digit_level - 1)), grid=(gsz // lsz, 1, 1), block=(lsz, 1, 1))
        return

    def Mul_i_i(self,gpuMemA,gpuMemB):
        self.kernel_Mul_i_i(gpuMemA, gpuMemB, grid=(gsz2 // lsz2, 1, 1), block=(lsz2, 1, 1))
        return

    def DivN(self,gpuMemA):
        self.kernel_DivN(gpuMemA, np.uint32(1<<digit_level), grid=(gsz2 // lsz2, 1, 1), block=(lsz2, 1, 1))
        return

    def PreNegFMT(self,gpuMemA,gpuMemB):
        self.kernel_PreNegFMT(gpuMemA,gpuMemB, self.MODP_WnSqrt,
                                    np.uint32(1<<digit_level), grid=(gsz2 // lsz2, 1, 1), block=(lsz2, 1, 1))
        return

    def PostNegFMT(self,gpuMemA):
        self.kernel_PostNegFMT(gpuMemA, self.MODP_WnSqrt,
                                     np.uint32(1<<digit_level), grid=(gsz2 // lsz2, 1, 1), block=(lsz2, 1, 1))
        return

    def PosNeg_To_HiLo(self,gpuMemE,gpuMemA,gpuMemB):
        self.kernel_PosNeg_To_HiLo(gpuMemE, gpuMemA,gpuMemB, np.uint32(1 << digit_level),
                                    grid=(gsz2 // lsz2, 1, 1), block=(lsz2, 1, 1))
        return

    def PostFMT_DivN_HiLo(self,gpuMemE,gpuMemA,gpuMemB):
        self.kernel_PostFMT_DivN_HiLo(gpuMemE, gpuMemA, gpuMemB, np.uint32(1 << digit_level),
                                 self.MODP_WnSqrt,grid=(gsz2 // lsz2, 1, 1), block=(lsz2, 1, 1))
        return

    # FMTで畳み込み乗算の結果を得る
    # A,Bは計算過程で内部が破壊されているので注意
    def Convolution(self,A,B):
        A_Neg = drv.mem_alloc(4 * digitN) #4=sizeof(uint)
        B_Neg = drv.mem_alloc(4 * digitN)  # 4=sizeof(uint)
        E = drv.mem_alloc(4 * digitN * 2)  # 4=sizeof(uint)

        self.PreNegFMT(A,A_Neg)#負順回用
        self.PreNegFMT(B,B_Neg)#負順回用
        self.uFMT(A)
        self.uFMT(B)
        self.uFMT(A_Neg)
        self.uFMT(B_Neg)
        self.Mul_i_i(A,B)#Bに結果が入る
        self.Mul_i_i(A_Neg,B_Neg)#Bに結果が入る
        self.iFMT(B)
        self.iFMT(B_Neg)
        self.PostNegFMT(B_Neg)
        self.DivN(B)#FFTみたくNで最後割らないといけない
        self.DivN(B_Neg)
        self.PosNeg_To_HiLo(E,B,B_Neg)#eに結果が入る
        #PostFMT_DivN_HiLo(E,B,B_Neg)   # 上4行のかわりにこれをつかってもよい

        drv.DeviceAllocation.free(A_Neg)
        drv.DeviceAllocation.free(B_Neg)
        return E



#検算用。A,Bの配列から一つの多倍長整数を作る。pythonのint型は乗算にKaratsuba法を使っているのでちょっと速い
def CalcAB(A,B):
    Alist = A.tolist()
    Blist = B.tolist()
    r=2**32
    for i in range(digit_level):
        for j in range(2**(digit_level-1-i)):
            Alist[j] = Alist[j * 2] + Alist[j * 2 + 1] * r
            Blist[j] = Blist[j * 2] + Blist[j * 2 + 1] * r
        r = r * r
    return Alist[0] * Blist[0]

#検算用。Eから一つの多倍長整数を作る
def CalcE(E):
    Elist = E.tolist()
    r = 2 ** 32
    for i in range(digit_level + 1):
        for j in range(2 ** (digit_level - i)):
            Elist[j] = Elist[j * 2] + Elist[j * 2 + 1] * r
        r = r * r
    return Elist[0]

def InitializeAB():
    host_A = np.zeros(digitN).astype(np.uint32)
    for i in range(digitN):
        host_A[i] = np.random.randint(0, 2107483647) * 2 + np.random.randint(0, 1)  # ここが大きいと畳み込み後の結果がmod Pにおさまりきらなくなる
        host_A[i] = np.uint32(2107483647)

    host_B = np.zeros(digitN).astype(np.uint32)
    for i in range(digitN):
        host_B[i] = np.random.randint(0, 2107483647) * 2 + np.random.randint(0, 1)
        host_B[i] = np.uint32(2107483647)
    return host_A,host_B

def CreateWnSqrt(a,m,n_level):
    ans=a
    for i in range(n_level-digit_level-1):#負巡回で使うことを考え-1
        ans=ans*ans%m
    return ans

#E0,E1,E2を使ってgarnerでEを復元して繰り上がり処理。結果はGPU内
def Carrying(E0,E1,E2,E):
    gsz3 = 1 << (digit_level+1)
    lsz3 = min(gsz3, 256)
    kernel_garner(E0,E1,E2,E,np.uint32(digitN*2),
        grid=(gsz3 // lsz3, 1, 1),block=(lsz3, 1, 1))
    return



if __name__ == '__main__':
    #digit_level = 16 #これは上で指定
    #digitN = 1 << digit_level
    print("A,Bの要素数=",digitN)
    FMT_P0 = FMTClass(MODP=469762049 , MODP_WnSqrt=CreateWnSqrt(60733,469762049,26))
    FMT_P1 = FMTClass(MODP=1811939329, MODP_WnSqrt=CreateWnSqrt(59189,1811939329,26))
    FMT_P2 = FMTClass(MODP=2013265921, MODP_WnSqrt=CreateWnSqrt(52278,2013265921,27))

    host_E = np.zeros(digitN * 2).astype(np.uint32)#結果格納用
    E = drv.to_device(host_E)  # gpuメモリ確保。かならず0に初期化しておかないといけない

    print("初期値生成")
    host_A, host_B = InitializeAB()
    A0 = drv.to_device(host_A%np.uint32(FMT_P0.MODP)) # gpuメモリ確保&転送
    B0 = drv.to_device(host_B%np.uint32(FMT_P0.MODP)) # gpuメモリ確保&転送
    A1 = drv.to_device(host_A%np.uint32(FMT_P1.MODP))  # gpuメモリ確保&転送
    B1 = drv.to_device(host_B%np.uint32(FMT_P1.MODP))  # gpuメモリ確保&転送
    A2 = drv.to_device(host_A%np.uint32(FMT_P2.MODP))  # gpuメモリ確保&転送
    B2 = drv.to_device(host_B%np.uint32(FMT_P2.MODP))  # gpuメモリ確保&転送

    print("GPU計算開始")
    stime = time.time()
    E0 = FMT_P0.Convolution(A0, B0)
    E1 = FMT_P1.Convolution(A1, B1)
    E2 = FMT_P2.Convolution(A2, B2)

    # ここからgarner&繰り上がり、E0,E1,E2から復元繰り上げしEに出力
    Carrying(E0, E1, E2 , E)
    drv.Context.synchronize() #タスク待ち命令
    print("GPU計算終了",time.time()-stime,"sec")

    # 結果をGPU→CPU
    drv.memcpy_dtoh(host_E, E)

    # 検算
    print("検算開始")
    CPUres = CalcAB(host_A, host_B)
    GPUres = CalcE(host_E)

    if CPUres!=GPUres:
        print("not equal !!!!!!!")
        print(CPUres)
        print(GPUres)
    else:
        print("あってる")