
import numpy as np

# aの逆元を返す mod m
def modinv(a,m):
    b = m
    u = 1
    v = 0
    while (b) :
        t = a // b
        a -= t * b
        tmp=a
        a=b
        b=tmp
        u -= t * v
        tmp = u
        u = v
        v = tmp
    u %= m
    if u < 0:
        u += m
    return u

# bit逆順 uint型を想定
def bitRev(a):
    a = (a & 0x55555555) << 1 | (a & 0xAAAAAAAA) >> 1
    a = (a & 0x33333333) << 2 | (a & 0xCCCCCCCC) >> 2
    a = (a & 0x0F0F0F0F) << 4 | (a & 0xF0F0F0F0) >> 4
    a = (a & 0x00FF00FF) << 8 | (a & 0xFF00FF00) >> 8
    a = (a & 0x0000FFFF) << 16 | (a & 0xFFFF0000) >> 16
    return a >> (32 - digit_level)

#garnerのアルゴリズム、3変数ver
#64bit整数におさまらない可能性あり、pythonのint型を使用
def Garner(a_,b_,c_,ar_,br_,cr_):
    a = int(a_)
    b = int(b_)
    c = int(c_)
    ar = int(ar_)
    br = int(br_)
    cr = int(cr_)
    x = ar + (br - ar) * modinv(a,b) % b * a
    x = x + (cr - x % c) * modinv(a,c) % c * modinv(b,c) % c * a * b
    #x %= a*b*c #a<b<cのときこの処理は必要ない
    return x

class FMTClass():
    def __init__(self,MODP,MODP_Wn):
        self.MODP=np.uint32(MODP)
        self.MODP_Wn=np.uint32(MODP_Wn)

    # MODP剰余下でaのb乗を返す
    def ModExp(self,a, b):
        ans = np.uint64(1)
        aa = np.uint64(a)
        while (b != 0):
            if (b % 2 == 1):
                ans = ans * aa % np.uint64(self.MODP)
            aa = aa * aa % np.uint64(self.MODP)
            b //= 2
        return np.uint32(ans)

    def FMT(self,inputA):
        # まず配列をビット逆順
        arrayA = inputA[bitRev(np.uint32(range(digitN)))]
        # FMTループ
        for i in range(digit_level):
            loopCnt_Pow2 = 1 << i
            for idx in range(digitN // 2):
                t2 = idx % loopCnt_Pow2
                t0 = idx * 2 - t2
                t1 = t0 + loopCnt_Pow2
                arrayAt0 = arrayA[t0]
                arrayAt1 = arrayA[t1]
                w0 = self.ModExp(self.MODP_Wn, t2 * (1 << (digit_level - 1 - i)))
                w1 = np.uint32(np.uint64(arrayAt1) * np.uint64(w0) % np.uint64(self.MODP))
                arrayA[t1] = (arrayAt0 + self.MODP - w1) % self.MODP
                arrayA[t0] = (arrayAt0 + w1) % self.MODP
        return arrayA[:]

    def iFMT(self,inputA):
        # まず配列をビット逆順
        arrayA = inputA[bitRev(np.uint32(range(digitN)))]
        # FMTループ
        for i in range(digit_level):
            loopCnt_Pow2 = 1 << i
            for idx in range(digitN // 2):
                t2 = idx % loopCnt_Pow2
                t0 = idx * 2 - t2
                t1 = t0 + loopCnt_Pow2
                arrayAt0 = arrayA[t0]
                arrayAt1 = arrayA[t1]
                w0 = self.ModExp(self.MODP_Wn, digitN - t2 * (1 << (digit_level - 1 - i)))
                w1 = np.uint32(np.uint64(arrayAt1) * np.uint64(w0) % np.uint64(self.MODP))
                arrayA[t1] = (arrayAt0 + self.MODP - w1) % self.MODP
                arrayA[t0] = (arrayAt0 + w1) % self.MODP
        return arrayA[:]

    def DivN(self,inputA):
        digitN_r = modinv(digitN, int(self.MODP))  # digitNの逆元を計算
        ret = np.uint64(inputA[:]) * np.uint64(digitN_r) % np.uint64(self.MODP)
        return np.uint32(ret)

    #A*Bの畳み込み結果を返す
    def Convolution(self,A_,B_):
        A = self.FMT(A_%self.MODP)
        B = self.FMT(B_%self.MODP)
        C = np.uint64(A[:]) * np.uint64(B[:]) % np.uint64(self.MODP)  # 各要素どうし乗算
        C = np.uint32(C) #結果は32bit範囲内におさまってるので
        C = self.iFMT(C)  # 逆変換
        C = self.DivN(C)  # Nで割る
        return C

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

#検算用。Cから一つの多倍長整数を作る
def CalcC(C):
    Clist = C.tolist()
    r = 2 ** 32
    for i in range(digit_level):
        for j in range(2 ** (digit_level-1 - i)):
            Clist[j] = Clist[j * 2] + Clist[j * 2 + 1] * r
        r = r * r
    return Clist[0]

def CreateWn(a,m,n_level):
    ans=a
    for i in range(n_level-digit_level):
        ans=ans*ans%m
    return ans

#E0,E1,E2を使ってgarnerでEを復元して繰り上がり処理
def Carrying(C0,C1,C2):
    C = np.zeros(digitN).astype(np.uint32)
    upg = 0  # 繰り上がり分 python3のint型なのでオーバーフローの心配はない
    for i in range(digitN):
        g = Garner(FMT_P0.MODP, FMT_P1.MODP, FMT_P2.MODP, C0[i], C1[i], C2[i])
        g += upg
        C[i] = np.uint32(g % 4294967296)
        upg = g//4294967296
    return C


if __name__=="__main__":
    digit_level = 14
    digitN = 1 << digit_level

    FMT_P0 = FMTClass(MODP=469762049 , MODP_Wn=CreateWn(60733,469762049,26))
    FMT_P1 = FMTClass(MODP=1811939329, MODP_Wn=CreateWn(59189,1811939329,26))
    FMT_P2 = FMTClass(MODP=2013265921, MODP_Wn=CreateWn(52278,2013265921,27))

    # C=A*Bをしたい
    # A,Bの初期化。前半にランダムな値をいれる。あとは0
    A_ = np.zeros(digitN).astype(np.uint32)
    A_[0:digitN // 2] = np.uint32( np.random.randint(0, 2147483648, (digitN // 2)) * 2
                                   + np.random.randint(0, 2, (digitN // 2)))
    B_ = np.zeros(digitN).astype(np.uint32)
    B_[0:digitN // 2] = np.uint32(np.random.randint(0, 2147483648, (digitN // 2)) * 2
                                  + np.random.randint(0, 2, (digitN // 2)))

    #p=469762049でやる
    C0 = FMT_P0.Convolution(A_,B_)

    #p=1811939329でやる
    C1 = FMT_P1.Convolution(A_, B_)

    #p=2013265921でやる
    C2 = FMT_P2.Convolution(A_, B_)

    #一部結果表示
    print("畳み込みの結果 mod","469762049",
          "\t\t畳み込みの結果 mod","1811939329",
          "\t\t畳み込みの結果 mod", "2013265921",
          "\t\tGarnerで復元")
    for i in range(10):
        print("C[",i,"] =",C0[i],
              "  \t\t\t\t", "C[", i, "] =", C1[i],
              " \t\t\t\t", "C[", i, "] =", C2[i],
              "\t\t\t\t",Garner(FMT_P0.MODP,FMT_P1.MODP,FMT_P2.MODP,C0[i],C1[i],C2[i])
              )

    #繰り上がり
    C=Carrying(C0,C1,C2)

    #検算
    print("検算開始")
    if CalcAB(A_, B_)==CalcC(C):
        print("正解！")
    else:
        print("ちがう！")