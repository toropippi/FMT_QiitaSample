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
    def __init__(self,MODP,MODP_WnSqrt):
        self.MODP=np.uint32(MODP)
        self.MODP_WnSqrt = np.uint32(MODP_WnSqrt)
        self.MODP_Wn = np.uint32(np.uint64(self.MODP_WnSqrt) * np.uint64(self.MODP_WnSqrt) % np.uint64(self.MODP))

    # MODP剰余下でaのb乗を返す
    # 追記1:np.uint32(pow(np.uint32(a),np.uint32(b),np.uint32(self.MODP)))を使ったほうが速い!
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
        arrayA=inputA.copy()
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
        arrayA = inputA.copy()
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

    def uFMT(self,inputA):
        arrayA = inputA.copy()
        for i in range(digit_level):
            loopCnt_Pow2 = 1 << (digit_level - i - 1)
            for idx in range(digitN // 2):
                t2 = idx % loopCnt_Pow2
                t0 = idx * 2 - t2
                t1 = t0 + loopCnt_Pow2
                arrayAt0 = arrayA[t0]
                arrayAt1 = arrayA[t1]
                w0 = self.ModExp(self.MODP_Wn, t2 * (1 << i))
                r0 = (arrayAt0 - arrayAt1 + self.MODP) % self.MODP
                r1 = (arrayAt0 + arrayAt1) % self.MODP
                w1 = np.uint32(np.uint64(r0) * np.uint64(w0) % np.uint64(self.MODP))
                arrayA[t1] = w1
                arrayA[t0] = r1
        return arrayA[:]

    def iuFMT(self,inputA):
        arrayA = inputA.copy()
        for i in range(digit_level):
            loopCnt_Pow2 = 1 << (digit_level - i - 1)
            for idx in range(digitN // 2):
                t2 = idx % loopCnt_Pow2
                t0 = idx * 2 - t2
                t1 = t0 + loopCnt_Pow2
                arrayAt0 = arrayA[t0]
                arrayAt1 = arrayA[t1]
                w0 = self.ModExp(self.MODP_Wn, digitN - t2 * (1 << i))
                r0 = (arrayAt0 - arrayAt1 + self.MODP) % self.MODP
                r1 = (arrayAt0 + arrayAt1) % self.MODP
                w1 = np.uint32(np.uint64(r0) * np.uint64(w0) % np.uint64(self.MODP))
                arrayA[t1] = w1
                arrayA[t0] = r1
        return arrayA[:]

    def DivN(self,inputA):
        digitN_r = modinv(digitN, int(self.MODP))  # digitNの逆元を計算
        ret = np.uint64(inputA[:]) * np.uint64(digitN_r) % np.uint64(self.MODP)
        return np.uint32(ret)

    def PreNegFMT(self,A):
        for i in range(digitN):
            A[i] = np.uint32(np.uint64(A[i]) * np.uint64(self.ModExp(self.MODP_WnSqrt, i)) % np.uint64(self.MODP))
        return A[:]

    def PostNegFMT(self,A):
        for i in range(digitN):
            A[i] = np.uint32(
                np.uint64(A[i]) * np.uint64(self.ModExp(self.MODP_WnSqrt, digitN * 2 - i)) % np.uint64(self.MODP))
        return A[:]

    #A*Bの畳み込み結果を返す
    #上位桁、下位桁を復元して2倍の要素数で返す
    def Convolution(self,A_,B_):
        #正巡回
        A = self.uFMT(A_%self.MODP)
        B = self.uFMT(B_%self.MODP)
        Cpos = np.uint64(A[:]) * np.uint64(B[:]) % np.uint64(self.MODP)  # 各要素どうし乗算
        Cpos = np.uint32(Cpos) #結果は32bit範囲内におさまってるので
        Cpos = self.iFMT(Cpos)  # 逆変換
        Cpos = self.DivN(Cpos)  # Nで割る

        # 負巡回
        A = self.PreNegFMT(A_ % self.MODP)#負巡回の前処理
        B = self.PreNegFMT(B_ % self.MODP)#負巡回の前処理
        A = self.uFMT(A)
        B = self.uFMT(B)
        Cneg = np.uint64(A[:]) * np.uint64(B[:]) % np.uint64(self.MODP)  # 各要素どうし乗算
        Cneg = np.uint32(Cneg)  # 結果は32bit範囲内におさまってるので
        Cneg = self.iFMT(Cneg)  # 逆変換
        Cneg = self.DivN(Cneg)  # Nで割る
        Cneg = self.PostNegFMT(Cneg)  # 負巡回の後処理

        #上位桁、下位桁 復元
        Elo = (Cpos + Cneg) % self.MODP
        Elo = (Elo + Elo % 2 * self.MODP) // 2

        Ehi = (Cpos - Cneg + self.MODP) % self.MODP
        Ehi = (Ehi + Ehi % 2 * self.MODP) // 2

        E = np.zeros(digitN*2).astype(np.uint32)
        E[:digitN] = Elo[:]
        E[digitN:] = Ehi[:]
        return E[:]

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

def CreateWnSqrt(a,m,n_level):
    ans=a
    for i in range(n_level-digit_level-1):#負巡回で使うことを考え-1
        ans=ans*ans%m
    return ans

#E0,E1,E2を使ってgarnerでEを復元して繰り上がり処理
def Carrying(E0,E1,E2):
    E = np.zeros(digitN * 2).astype(np.uint32)
    upg = 0  # 繰り上がり分 python3のint型なのでオーバーフローの心配はない
    for i in range(digitN * 2):
        g = Garner(FMT_P0.MODP, FMT_P1.MODP, FMT_P2.MODP, E0[i], E1[i], E2[i])
        g += upg
        E[i] = np.uint32(g % 4294967296)
        upg = g//4294967296
    return E



if __name__=="__main__":
    digit_level = 16
    digitN = 1 << digit_level

    FMT_P0 = FMTClass(MODP=469762049 , MODP_WnSqrt=CreateWnSqrt(60733,469762049,26))
    FMT_P1 = FMTClass(MODP=1811939329, MODP_WnSqrt=CreateWnSqrt(59189,1811939329,26))
    FMT_P2 = FMTClass(MODP=2013265921, MODP_WnSqrt=CreateWnSqrt(52278,2013265921,27))

    # A,Bの初期化。前半後半ともにランダムな値をいれる。
    A_ = np.zeros(digitN).astype(np.uint32)
    A_[:] = np.uint32( np.random.randint(0, 2147483648, (digitN)) * 2
                                   + np.random.randint(0, 2, (digitN)))
    B_ = np.zeros(digitN).astype(np.uint32)
    B_[:] = np.uint32(np.random.randint(0, 2147483648, (digitN)) * 2
                                  + np.random.randint(0, 2, (digitN)))

    #p=469762049でやる
    E0 = FMT_P0.Convolution(A_,B_)

    #p=1811939329でやる
    E1 = FMT_P1.Convolution(A_, B_)

    #p=2013265921でやる
    E2 = FMT_P2.Convolution(A_, B_)

    #結果一部表示表示
    print("畳み込みの結果 mod","469762049",
          "\t\t畳み込みの結果 mod","1811939329",
          "\t\t畳み込みの結果 mod", "2013265921",
          "\t\tGarnerで復元")
    for i in range(10):
        print("E[",i,"] =",E0[i],
              "  \t\t\t\t", "E[", i, "] =", E1[i],
              " \t\t\t\t", "E[", i, "] =", E2[i],
              "\t\t\t\t",Garner(FMT_P0.MODP,FMT_P1.MODP,FMT_P2.MODP,E0[i],E1[i],E2[i])
              )
    #繰り上がり
    E=Carrying(E0,E1,E2)

    #検算
    print("検算開始")
    if CalcAB(A_, B_)==CalcE(E):
        print("正解！")
    else:
        print("ちがう！")