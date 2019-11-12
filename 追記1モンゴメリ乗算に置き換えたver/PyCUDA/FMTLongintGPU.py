# FMT(高速剰余変換) PyCUDA版
# 3つの素数PのもとでFMTを行うサンプル
# garnerのアルゴリズムで最後に値を復元しているため、2の90.7乗まで正確に復元できる
# 検算関数もあり
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import os


programid_garner=SourceModule("""
    #include "garner.cu"
""",include_dirs=[os.getcwd()])
kernel_garner=programid_garner.get_function("GarnerGPU")

programid_addsub=SourceModule("""
    #include "addsubmod.cu"
""",include_dirs=[os.getcwd()])
kernel_CreateAmod=programid_addsub.get_function("CreateAmod")
kernel_SubCAB=programid_addsub.get_function("SubCAB")
kernel_AddCAB=programid_addsub.get_function("AddCAB")
kernel_SetZero=programid_addsub.get_function("SetZero")
kernel_ShiftBA=programid_addsub.get_function("ShiftBA")
kernel_memcpy_dtod_ui4=programid_addsub.get_function("memcpy_dtod_ui4")
kernel_FirstNewtonRev=programid_addsub.get_function("FirstNewtonRev")
kernel_FirstNewtonSqrt=programid_addsub.get_function("FirstNewtonSqrt")





class FMTClass():
    def __init__(self,MODP):
        self.MODP=np.uint32(MODP)
        #self.MODP_WnSqrt = np.uint32(MODP_WnSqrt)
        #self.MODP_Wn = np.uint32(np.uint64(self.MODP_WnSqrt) * np.uint64(self.MODP_WnSqrt) % np.uint64(self.MODP))
        self.MODP_WnSqrt = np.uint32(1)
        self.MODP_Wn = np.uint32(1)
        if False:
            self.programid = SourceModule("""
                #define MODP (uint)(""" + str(MODP) + """)
                    #include "fmt.cu"
                """, include_dirs=[os.getcwd()])
        if MODP==2013265921:
            self.programid = SourceModule("""
                #define MODP (uint)(""" + str(MODP) + """)
                #define rMODP (uint)(2290649223)
                #define MODLOG (uint)(31)
                    #include "fmt.cu"
                """, include_dirs=[os.getcwd()])

        if MODP==1811939329:
            self.programid = SourceModule("""
                #define MODP (uint)(""" + str(MODP) + """)
                #define rMODP (uint)(2545165803)
                #define MODLOG (uint)(31)
                    #include "fmt.cu"
                """, include_dirs=[os.getcwd()])

        if MODP==469762049:
            self.programid = SourceModule("""
                #define MODP (uint)(""" + str(MODP) + """)
                #define rMODP (uint)(613566755)
                #define MODLOG (uint)(29)
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

    def set_digit_level(self,level):
        self.digit_level = level  #################################可変パラメーター############################# 最大25
        self.digitN = 1 << self.digit_level
        self.gsz = 1 << (self.digit_level - 1)  # gpu global_work_size
        self.lsz = min(self.gsz, 256)  # gpu local_work_size
        self.gsz2 = self.digitN
        self.lsz2 = min(self.gsz2, 256)
        return

    def CreateWnSqrt(self,a, m, n_level):
        ans = a
        for i in range(n_level - self.digit_level - 1):  # 負巡回で使うことを考え-1
            ans = ans * ans % m
        self.MODP_WnSqrt=np.uint32(ans)
        self.MODP_Wn = np.uint32(np.uint64(self.MODP_WnSqrt) * np.uint64(self.MODP_WnSqrt) % np.uint64(self.MODP))
        return

    def FMT(self,gpuMemA):
        for i in range(self.digit_level):
            self.kernel_FMT(gpuMemA, np.uint32(1 << i), self.MODP_Wn, np.uint32(1 << (self.digit_level - 1)),
                         grid=(self.gsz // self.lsz, 1, 1), block=(self.lsz, 1, 1))
        return

    def uFMT(self,gpuMemA):
        for i in range(self.digit_level):
            self.kernel_uFMT(gpuMemA, np.uint32((1<<(self.digit_level-1))>>i), self.MODP_Wn, np.uint32(1 << (self.digit_level - 1)),
                         grid=(self.gsz // self.lsz, 1, 1), block=(self.lsz, 1, 1))
        return

    def iFMT(self,gpuMemA):
        for i in range(self.digit_level):
            self.kernel_iFMT(gpuMemA, np.uint32(1 << i), self.MODP_Wn,
                         np.uint32(1 << (self.digit_level - 1)), grid=(self.gsz // self.lsz, 1, 1), block=(self.lsz, 1, 1))
        return

    def iuFMT(self,gpuMemA):
        for i in range(self.digit_level):
            self.kernel_iuFMT(gpuMemA, np.uint32(1<<(self.digit_level-1-i)), self.MODP_Wn,
                         np.uint32(1 << (self.digit_level - 1)), grid=(self.gsz // self.lsz, 1, 1), block=(self.lsz, 1, 1))
        return

    def Mul_i_i(self,gpuMemA,gpuMemB):
        self.kernel_Mul_i_i(gpuMemA, gpuMemB, grid=(self.gsz2 // self.lsz2, 1, 1), block=(self.lsz2, 1, 1))
        return

    def DivN(self,gpuMemA):
        self.kernel_DivN(gpuMemA, np.uint32(1<<self.digit_level), grid=(self.gsz2 // self.lsz2, 1, 1), block=(self.lsz2, 1, 1))
        return

    def PreNegFMT(self,gpuMemA,gpuMemB):
        self.kernel_PreNegFMT(gpuMemA,gpuMemB, self.MODP_WnSqrt,
                                    np.uint32(1<<self.digit_level), grid=(self.gsz2 // self.lsz2, 1, 1), block=(self.lsz2, 1, 1))
        return

    def PostNegFMT(self,gpuMemA):
        self.kernel_PostNegFMT(gpuMemA, self.MODP_WnSqrt,
                                     np.uint32(1<<self.digit_level), grid=(self.gsz2 // self.lsz2, 1, 1), block=(self.lsz2, 1, 1))
        return

    def PosNeg_To_HiLo(self,gpuMemE,gpuMemA,gpuMemB):
        self.kernel_PosNeg_To_HiLo(gpuMemE, gpuMemA,gpuMemB, np.uint32(1 << self.digit_level),
                                    grid=(self.gsz2 // self.lsz2, 1, 1), block=(self.lsz2, 1, 1))
        return

    def PostFMT_DivN_HiLo(self,gpuMemE,gpuMemA,gpuMemB):
        self.kernel_PostFMT_DivN_HiLo(gpuMemE, gpuMemA, gpuMemB, np.uint32(1 << self.digit_level),
                                 self.MODP_WnSqrt,grid=(self.gsz2 // self.lsz2, 1, 1), block=(self.lsz2, 1, 1))
        return

    # FMTで畳み込み乗算の結果を得る
    # A,Bは計算過程で内部が破壊されているので注意
    def Convolution(self,A,B):
        A_Neg = drv.mem_alloc(4 * self.digitN) #4=sizeof(uint)
        B_Neg = drv.mem_alloc(4 * self.digitN)  # 4=sizeof(uint)
        E = drv.mem_alloc(4 * self.digitN * 2)  # 4=sizeof(uint)

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
        #drv.DeviceAllocation.free(A_Neg)
        #drv.DeviceAllocation.free(B_Neg)
        return E



#E0,E1,E2を使ってgarnerでEを復元して繰り上がり処理。結果はGPU内
def Carrying(E0,E1,E2,E,digit_level):
    digitN = 1 << digit_level
    gsz3 = 1 << (digit_level+1)
    lsz3 = min(gsz3, 256)
    kernel_garner(E0,E1,E2,E,np.uint32(digitN*2),
        grid=(gsz3 // lsz3, 1, 1),block=(lsz3, 1, 1))
    return

















FMT_P0 = FMTClass(MODP=469762049 )
FMT_P1 = FMTClass(MODP=1811939329)
FMT_P2 = FMTClass(MODP=2013265921)

# 外部から使いたい関数
# 乗算するABは同じlevelでないといけない。
# levelに応じてWnが変わる
# E=A*B
def Mulfunc(inA,inB):
    digit_level=inA.intsize_level
    digitN = 1 << digit_level
    gsz = digitN
    lsz = min(gsz, 256)
    gsz2 = digitN * 2
    lsz2 = min(gsz2, 256)

    FMT_P0.set_digit_level(digit_level)
    FMT_P1.set_digit_level(digit_level)
    FMT_P2.set_digit_level(digit_level)
    FMT_P0.CreateWnSqrt(60733, 469762049, 26)
    FMT_P1.CreateWnSqrt(59189, 1811939329, 26)
    FMT_P2.CreateWnSqrt(52278, 2013265921, 27)

    E = drv.mem_alloc(digitN * 2 * 4)
    kernel_SetZero(E, grid=(gsz2 // lsz2, 1, 1), block=(lsz2, 1, 1))
    A0 = drv.mem_alloc(digitN * 4)
    A1 = drv.mem_alloc(digitN * 4)
    A2 = drv.mem_alloc(digitN * 4)
    B0 = drv.mem_alloc(digitN * 4)
    B1 = drv.mem_alloc(digitN * 4)
    B2 = drv.mem_alloc(digitN * 4)

    kernel_CreateAmod(A0, inA.number, np.uint32(469762049), grid=(gsz // lsz, 1, 1),
                      block=(lsz, 1, 1))
    kernel_CreateAmod(A1, inA.number, np.uint32(1811939329), grid=(gsz // lsz, 1, 1),
                      block=(lsz, 1, 1))
    kernel_CreateAmod(A2, inA.number, np.uint32(2013265921), grid=(gsz // lsz, 1, 1),
                      block=(lsz, 1, 1))
    kernel_CreateAmod(B0, inB.number, np.uint32(469762049), grid=(gsz // lsz, 1, 1),
                      block=(lsz, 1, 1))
    kernel_CreateAmod(B1, inB.number, np.uint32(1811939329), grid=(gsz // lsz, 1, 1),
                      block=(lsz, 1, 1))
    kernel_CreateAmod(B2, inB.number, np.uint32(2013265921), grid=(gsz // lsz, 1, 1),
                      block=(lsz, 1, 1))

    E0 = FMT_P0.Convolution(A0, B0)
    E1 = FMT_P1.Convolution(A1, B1)
    E2 = FMT_P2.Convolution(A2, B2)

    # ここからgarner&繰り上がり、E0,E1,E2から復元繰り上げしEに出力
    Carrying(E0, E1, E2, E, digit_level)
    return E

# 外部から使いたい関数
# 乗算するABは同じlevelでないといけない。
#C=A+B
def Addfunc(inA,inB):
    digit_level=inA.intsize_level
    digitN = 1 << digit_level
    gsz = digitN
    lsz = min(gsz, 256)
    C = drv.mem_alloc(digitN * 4)
    kernel_SetZero(C, grid=(gsz // lsz, 1, 1), block=(lsz, 1, 1))
    kernel_AddCAB(C,inA.number, inB.number, np.uint32(digitN), grid=(gsz // lsz, 1, 1),
                      block=(lsz, 1, 1))
    return C

# 外部から使いたい関数
# 乗算するABは同じlevelでないといけない。
#C=A-B
def Subfunc(inA,inB):
    digit_level=inA.intsize_level
    digitN = 1 << digit_level
    gsz = digitN
    lsz = min(gsz, 256)
    C = drv.mem_alloc(digitN * 4)
    kernel_SetZero(C, grid=(gsz // lsz, 1, 1), block=(lsz, 1, 1))
    kernel_SubCAB(C,inA.number, inB.number, np.uint32(digitN), grid=(gsz // lsz, 1, 1),
                      block=(lsz, 1, 1))
    return C


# 外部から使いたい関数
# shift
#C=A<<n
def BitShiftfunc(inA,n):
    digit_level=inA.intsize_level
    digitN = 1 << digit_level
    gsz = digitN
    lsz = min(gsz, 256)
    B = drv.mem_alloc(digitN * 4)
    kernel_SetZero(B, grid=(gsz // lsz, 1, 1), block=(lsz, 1, 1))
    kernel_ShiftBA(B,inA.number,np.int32(n),np.uint32(digitN), grid=(gsz // lsz, 1, 1),
                      block=(lsz, 1, 1))
    return B



# 外部から使いたい関数
# memcpy
# uint限定、非同期ver
# copysizeはbyte数ではなくuintの要素数を指定
def memcpy_dtod_ui4(inB,inA,boffset,aoffset,copysize):
    c=(copysize+255)//256
    gsz = c*256
    lsz = min(gsz, 256)
    kernel_memcpy_dtod_ui4(inB,inA,np.uint32(boffset),np.uint32(aoffset),np.uint32(copysize)
                           , grid=(gsz // lsz, 1, 1),block=(lsz, 1, 1))
    return

def FirstNewtonRev(inA):
    ret = drv.mem_alloc(2 * 4)
    kernel_FirstNewtonRev(ret,inA.number,np.uint32(inA.digitN-1), grid=(1, 1, 1),block=(1, 1, 1))
    return ret

def FirstNewtonSqrt(inA):
    ret = drv.mem_alloc(2 * 4)
    kernel_FirstNewtonSqrt(ret, inA.number, np.uint32(inA.digitN - 1), grid=(1, 1, 1), block=(1, 1, 1))
    return ret