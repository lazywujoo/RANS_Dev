class input:
    def __init__(self,inp):
        self.nz = 200
        self.Uinf = inp[0]
        self.utau = inp[1]
        self.zpls2= 0.2 
        self.ztau = inp[2]
        self.chi = 1.025
        self.delta_Ref = 0.0353 
        self.Tw = inp[3]
        self.Tinf = inp[4]
        self.gam = 1.4
        self.Minf = inp[5]
        self.rinf = inp[6]
        self.L  = inp[10]/9.4
        self.igmres = 1
        self.itempeq = 1
        self.iRANS = 1 # 1 = SST 2 = Cess 3 = v2f
        self.rhow = inp[7]
        self.delta = inp[8]
        self.Retau = inp[9]