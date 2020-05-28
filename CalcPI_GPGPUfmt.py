# 円周率piをGPUで計算するコード
# digit_levelを変えることで出力桁数を変えられる。後は変えられるパラメータはない
import FMTLongintGPU as flint

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import time


cardinal = 4294967296 # 基数
digit_level = 25 #2以下または26以上で正しく実行できない
#digit_level 20のとき500万桁(10進数)
#21で1000万、24で8000万桁
print(cardinal,"進数",(1<<digit_level),"桁")



class GPULongint():
    def __init__(self):
        self.hide_digit = 0  # 隠れ桁数。実際の値のほうが大きい場合はこれがプラスになる
        self.intsize_level = 0  # longint型が保持しているサイズのlog2n。fmtで使うdigit_levelと同じ
        self.digitN = 1 << self.intsize_level
        self.number = 0 #vram
        return

    def mul(self, a):  # self * a ただしintsize_level不変
        e_number=flint.Mulfunc(self,a) # eの桁数は2倍
        self.hide_digit = self.hide_digit + a.hide_digit + (1 << a.intsize_level)
        oute = drv.mem_alloc(4 * self.digitN)
        flint.memcpy_dtod_ui4(oute,e_number, 0, self.digitN,self.digitN)
        self.number = oute
        return

    #1桁ずらして出力
    def mull1(self, a):  # self * a ただしintsize_level不変
        e_number=flint.Mulfunc(self,a) # eの桁数は2倍
        self.hide_digit = self.hide_digit + a.hide_digit + (1 << a.intsize_level) - 1
        oute = drv.mem_alloc(4 * self.digitN)
        flint.memcpy_dtod_ui4(oute,e_number, 0, self.digitN-1,self.digitN)
        self.number = oute
        return

    def copy(self):
        ret = GPULongint()
        ret.hide_digit = self.hide_digit
        ret.intsize_level = self.intsize_level
        ret.digitN = 1 << self.intsize_level
        dmy = drv.mem_alloc(4 * self.digitN)
        drv.memcpy_dtod_async(dmy, self.number, 4 * self.digitN)
        ret.number = dmy
        return ret

    #self=a+b
    def add_abc(self, a, b):
        if b.hide_digit != a.hide_digit:
            if b.hide_digit < a.hide_digit:
                a.BitShift((a.hide_digit - b.hide_digit)*32)
                a.hide_digit -= a.hide_digit - b.hide_digit
            else:
                b.BitShift((b.hide_digit - a.hide_digit)*32)
                b.hide_digit -= b.hide_digit - a.hide_digit

        c_number=flint.Addfunc(a,b)
        self.number = c_number
        self.hide_digit = a.hide_digit
        self.intsize_level = a.intsize_level
        self.digitN = a.digitN
        return

    def BitShift(self,shift_n):
        #shift_nだけGPU上でビットシフト。マイナスもいける
        self.number = flint.BitShiftfunc(self, shift_n)
        return

    # self=self-a
    def sub_aba(self, a):
        if self.hide_digit != a.hide_digit:
            if self.hide_digit < a.hide_digit:
                a.BitShift((a.hide_digit - self.hide_digit)*32)
                a.hide_digit -= a.hide_digit - self.hide_digit
            else:
                self.BitShift((self.hide_digit - a.hide_digit)*32)
                self.hide_digit -= self.hide_digit - a.hide_digit

        c_number=flint.Subfunc(self, a)
        self.number = c_number
        return

    # self=a-self
    def sub_abb(self, a):
        if self.hide_digit != a.hide_digit:
            if self.hide_digit < a.hide_digit:
                a.BitShift((a.hide_digit - self.hide_digit)*32)
                a.hide_digit -= a.hide_digit - self.hide_digit
            else:
                self.BitShift((self.hide_digit - a.hide_digit)*32)
                self.hide_digit -= self.hide_digit - a.hide_digit

        c_number=flint.Subfunc(a, self)
        self.number = c_number
        return


    #自分が保持している値を上位桁にして、0埋めの下位桁を追加。メモリサイズはちょうど2倍になる
    def ReAlloc_Up(self):
        newnum = drv.mem_alloc(4 * self.digitN * 2)
        gsz = self.digitN * 2
        lsz = min(gsz, 256)
        flint.kernel_SetZero(newnum, grid=(gsz // lsz, 1, 1), block=(lsz, 1, 1))
        flint.memcpy_dtod_ui4(newnum, self.number, self.digitN, 0, self.digitN)
        self.number=newnum
        self.hide_digit -= 1 << self.intsize_level
        self.intsize_level += 1
        self.digitN *= 2
        return


    # GPULongint型のselfから上位1<<level桁を抽出して返す
    # オーバーフローに注意
    def PickPartial_Up(self,level):
        dN=1<<level
        newnum = drv.mem_alloc(4 * dN)
        flint.memcpy_dtod_ui4(newnum, self.number,0,self.digitN-dN,dN)
        ret=GPULongint()
        ret.number=newnum
        ret.hide_digit = self.hide_digit + (self.digitN - dN)
        ret.intsize_level = level
        ret.digitN = 1 << level
        return ret

    # return 1.0/self
    def NewtonReverse(self):
        GPUlongintx = GPULongint()
        GPUlongintx.number=flint.FirstNewtonRev(self)
        GPUlongintx.hide_digit = -1
        GPUlongintx.intsize_level = 1
        GPUlongintx.digitN = 1 << GPUlongintx.intsize_level

        for i in range(self.intsize_level):
            GPULongintA=self.PickPartial_Up(i+1)
            if i >= 1:
                GPUlongintx.ReAlloc_Up()
            GPUlongintR0 = GPULongintA.copy()
            GPUlongintR0.mul(GPUlongintx)
            GPUlongintR0.mull1(GPUlongintx)
            GPUlongintx.BitShift(1)
            GPUlongintx.sub_aba(GPUlongintR0)

        GPUlongintR0 = GPULongintA.copy()
        GPUlongintR0.mul(GPUlongintx)
        GPUlongintR0.mull1(GPUlongintx)
        GPUlongintx.BitShift(1)
        GPUlongintx.sub_aba(GPUlongintR0)
        return GPUlongintx



    def NewtonSqrt(self):
        GPUlongint3=GPUlongint3_.copy()
        GPUlongintx = GPULongint()
        GPUlongintx.number=flint.FirstNewtonSqrt(self)
        GPUlongintx.hide_digit = -1
        GPUlongintx.intsize_level = 1
        GPUlongintx.digitN = 1 << GPUlongintx.intsize_level

        for i in range(self.intsize_level):
            GPULongintA=self.PickPartial_Up(i+1)
            if i >= 1:
                GPUlongintx.ReAlloc_Up()
            GPUlongint3.ReAlloc_Up()
            GPUlongintR0=GPULongintA.copy()
            if i == 0:
                GPUlongintR0.mull1(GPUlongintx)
            else:
                GPUlongintR0.mul(GPUlongintx)
            GPUlongintR0.mul(GPUlongintx)
            GPUlongintR0.sub_abb(GPUlongint3)
            GPUlongintR0.mull1(GPUlongintx)
            GPUlongintR0.BitShift(-1)
            GPUlongintx = GPUlongintR0.copy()

        GPUlongintR0 = GPULongintA.copy()
        GPUlongintR0.mul(GPUlongintx)
        GPUlongintR0.mul(GPUlongintx)
        GPUlongintR0.sub_abb(GPUlongint3)
        GPUlongintR0.mull1(GPUlongintx)
        GPUlongintR0.BitShift(-1)
        GPUlongintx = GPUlongintR0.copy()
        GPUlongintx.mull1(GPULongintA)
        return GPUlongintx




def Create3():
    #3をつくるだけ
    h = np.zeros(1).astype(np.uint32)
    h[0]=np.uint32(3)
    gpuh=drv.to_device(h)
    GPUlongint3_ = GPULongint()
    GPUlongint3_.hide_digit = 0
    GPUlongint3_.intsize_level = 0
    GPUlongint3_.digitN = 1 << GPUlongint3_.intsize_level
    GPUlongint3_.number=gpuh
    return GPUlongint3_

#初期値生成
def ABTPinit():
    #a=1.0
    #b=sqrt(0.5)
    #t=0.25
    #p=1
    #ただしa=1.0は最初桁数的にオーバーフローするので、最初の1ループ目で使うaがらみの計算だけは特殊処理する

    # a=1.0
    a = GPULongint()
    a.hide_digit = -(1<<digit_level)
    a.intsize_level = digit_level
    a.digitN = 1 << a.intsize_level
    newnum = drv.mem_alloc(4 * a.digitN)
    gsz = a.digitN
    lsz = min(gsz, 256)
    flint.kernel_SetZero(newnum, grid=(gsz // lsz, 1, 1), block=(lsz, 1, 1))
    a.number = newnum

    # a_half=0.5
    h = np.zeros(1).astype(np.uint32)
    h[0]=np.uint32(1 << 31)

    a_half = GPULongint()
    a_half.hide_digit = -(1<<digit_level)
    a_half.intsize_level = digit_level
    a_half.digitN = 1 << a.intsize_level

    gpuh=drv.to_device(h)
    newnum = drv.mem_alloc(4 * a.digitN)
    flint.kernel_SetZero(newnum, grid=(gsz // lsz, 1, 1), block=(lsz, 1, 1))
    flint.memcpy_dtod_ui4(newnum, gpuh, a.digitN - 1, 0, 1) #最上位桁に0.5を代入
    a_half.number=newnum

    # b=sqrt(0.5)
    b = a_half.copy()
    b = b.NewtonSqrt()

    # t=0.25
    t = a_half.copy() #0.5
    t.BitShift(-1)# =0.25

    #p=1 ループごとに2倍になる
    p=1
    pp=0 #ループごとに1増える
    return a,a_half,b,t,p,pp







if __name__=="__main__":
    #初期値
    print("開始")
    st=time.time()
    stall=st

    GPUlongint3_=Create3()#sqrtで使う
    a,a_half,b,t,p,pp=ABTPinit()
    checkHost=np.zeros(a.digitN).astype(np.uint32)

    drv.Context.synchronize()

    print("初期化",time.time()-st,"sec")
    st=time.time()



    #ガウスルジャンドルのループ
    for i in range(digit_level*2+5):
        # anew=(a+b)//2 #############
        if i!=0:
            a_half = a.copy()
            a_half.BitShift(-1)
        b_half = b.copy()
        b_half.BitShift(-1)

        anew=a.copy()
        anew.add_abc(a_half,b_half)
        #ここまで #############

        # b=sqrt(a*b) #############
        ab=b.copy()
        if i!=0:
            ab.mul(a)#初回のみaの大きさ注意
        b = ab.NewtonSqrt()
        # ここまで #############


        # t=t-p*(a-anew)*(a-anew)//digit #############
        a.sub_aba(anew) #初回のみaの大きさ注意、これに関してはうまくアンダーフローしてくれるので結果的に大丈夫
        aa2=a.copy()
        aa2.mul(a) #(a-anew)*(a-anew)
        aa2.BitShift(pp) #aa2.number*=p
        t.sub_aba(aa2) #t=t-aa2
        #ここまで #############

        p*=2#これは結果つかわない・・
        pp+=1
        a=anew.copy()

        if i>digit_level:#ループ抜け判定
            drv.memcpy_dtoh(checkHost,aa2.number)
            flag=0
            for j in range(aa2.digitN):
                if checkHost[j]!=0:
                    flag=1
                    break
            if flag==0:
                break


    drv.Context.synchronize()
    print("ガウスルジャンドルループ",time.time()-st,"sec")
    st=time.time()

    #最後の (a+b)*(a+b)/4t
    tr=t.NewtonReverse()
    a_half=a.copy()
    b_half=b.copy()
    a_half.BitShift(-1)  #//= 2
    b_half.BitShift(-1)  #//= 2
    anew=a_half.copy()
    anew.add_abc(a_half, b_half) #anew=(a+b)/2
    ab=anew.copy()
    ab.mul(anew) #=anew*anew
    ab.mul(tr) # ab/=t
    drv.Context.synchronize()#GPUタスクまち命令

    print("最後の計算",time.time()-st,"sec")
    print("全体で",time.time()-stall,"sec")


    ans=np.zeros(ab.digitN).astype(np.uint32)
    drv.memcpy_dtoh(ans,ab.number)
    for i in range(min(ab.digitN,40)): #一部表示
        print("{:08X}".format(ans[ab.digitN-i-1]))

    # 出力
    path_w = 'outpi.txt'
    s = 'New file'
    with open(path_w, mode='w') as f:
        f.write("3.")
        for i in range(ab.digitN-1):  # 一部表示
            f.write("{:08X}".format(ans[ab.digitN - i - 2]))
    print("outpi.txtに出力しました",(1<<digit_level)/131072,"MB")
    #※最後のほうは合ってないですが仕様です