import numpy as np

# MODP剰余下でaのb乗を返す
def ModExp(a, b):
    ans = np.uint64(1)
    aa = np.uint64(a)
    while (b != 0):
        if (b % 2 == 1):
            ans = ans * aa % np.uint64(MODP)
        aa = aa * aa % np.uint64(MODP)
        b //= 2
    return np.uint32(ans)

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
    digitN_r = ModExp(digitN, MODP - 2)  # digitNのモジュラ逆数を計算
    ret = np.uint64(inputA[:]) * np.uint64(digitN_r) % np.uint64(MODP)
    return np.uint32(ret)


if __name__=="__main__":
    # 可変パラメーター
    digit_level = 3
    digitN = 1 << digit_level
    MODP = np.uint32(257)
    MODP_Wn = np.uint32(4) # 1の原始N乗根でなければならない。4の8乗=65536。65536%257=1

    # C=A*Bをしたい
    A=np.uint32([1,4,1,4,0,0,0,0])
    B=np.uint32([2,1,3,5,0,0,0,0])

    A = FMT(A)  # 変換
    B = FMT(B)  # 変換
    C = np.uint64(A[:]) * np.uint64(B[:]) % np.uint64(MODP)  # 各要素どうし乗算
    C = np.uint32(C)
    C = iFMT(C)  # 逆変換
    C = DivN(C)  # Nで割る

    print("畳み込みの結果")
    for i in range(digitN):
        print("C[",i,"] =",C[i])