import numpy as np

# MODP剰余下でaのb乗を返す
# 追記1:np.uint32(pow(np.uint32(a),np.uint32(b),np.uint32(MODP)))を使ったほうが速い!
def ModExp(a, b):
    ans = np.uint64(1)
    aa = np.uint64(a)
    while (b != 0):
        if (b % 2 == 1):
            ans = ans * aa % np.uint64(MODP)
        aa = aa * aa % np.uint64(MODP)
        b //= 2
    return np.uint32(ans)

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

def FMT(inputA):
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
            w0 = ModExp(MODP_Wn, t2 * (1 << (digit_level - 1 - i)))
            w1 = np.uint32(np.uint64(arrayAt1) * np.uint64(w0) % np.uint64(MODP))
            arrayA[t1] = (arrayAt0 + MODP - w1) % MODP
            arrayA[t0] = (arrayAt0 + w1) % MODP
    return arrayA[:]

def iFMT(inputA):
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
            w0 = ModExp(MODP_Wn, digitN - t2 * (1 << (digit_level - 1 - i)))
            w1 = np.uint32(np.uint64(arrayAt1) * np.uint64(w0) % np.uint64(MODP))
            arrayA[t1] = (arrayAt0 + MODP - w1) % MODP
            arrayA[t0] = (arrayAt0 + w1) % MODP
    return arrayA[:]

def DivN(inputA):
    digitN_r = modinv(digitN, int(MODP))  # digitNの逆元を計算
    ret = np.uint64(inputA[:]) * np.uint64(digitN_r) % np.uint64(MODP)
    return np.uint32(ret)


if __name__=="__main__":
    digit_level = 3
    digitN = 1 << digit_level

    # C=A*Bをしたい
    A_=np.uint32([9,9,9,9,0,0,0,0])
    B_=np.uint32([9,9,9,9,0,0,0,0])

    #まずp=257でやる
    MODP = np.uint32(257)
    MODP_Wn = np.uint32(4) # 1の原始N乗根でなければならない。4の8乗=65536。65536%257=1

    A = FMT(A_)  # 変換
    B = FMT(B_)  # 変換
    C0 = np.uint64(A[:]) * np.uint64(B[:]) % np.uint64(MODP)  # 各要素どうし乗算
    C0 = np.uint32(C0)
    C0 = iFMT(C0)  # 逆変換
    C0 = DivN(C0)  # Nで割る

    #p=17でやる
    MODP = np.uint32(17)
    MODP_Wn = np.uint32(2) # 1の原始N乗根でなければならない。2の8乗=256。256%17=1

    A = FMT(A_)  # 変換
    B = FMT(B_)  # 変換
    C1 = np.uint64(A[:]) * np.uint64(B[:]) % np.uint64(MODP)  # 各要素どうし乗算
    C1 = iFMT(C1)  # 逆変換
    C1 = DivN(C1)  # Nで割る

    print("畳み込みの結果 mod","257","\t\t畳み込みの結果 mod","17","\t\tGarnerで復元")
    for i in range(digitN):
        print("C[",i,"] =",C0[i],
              " \t\t\t\t","C[",i,"] =",C1[i],
              "\t\t\t\t",int(C0[i])+((int(C1[i])-int(C0[i]))*modinv(257,17))%17*257
              )