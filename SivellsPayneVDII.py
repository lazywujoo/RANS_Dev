import cupy as cp
from scipy.optimize import fsolve
import pdb 

# Reference : An evaluation of theories for predicting turbulent skin friction andheat transfer on flat plates at supersonic and hypersonic Mach numbers
# EDWARD J. HOPKINS and MAMORU INOUYE
# AIAA Journal 1971 9:6, 993-1003

file = "InputTable.dat"
Inputtab = cp.loadtxt(file, skiprows=1)
recov = 0.9
power = 0.76
# fluid property for air
rbar = 8314.3/28.96
Cp = 1.4*rbar/(1.4-1.)
# sample number
# increase n if Retau does not reach to a desired level
n = 10000
# opening a inputsave file
f= open("1D-HBL.inp", "w")
f.write('Uinf utau ztau Twall Tinf Minf rhoinf rhow delta Retau x ReL\n')
for icase in range(0, Inputtab.shape[0]):
    Tinf = Inputtab[icase,1]
    rinf = Inputtab[icase,3]
    Twall = Inputtab[icase,2]
    Minf = Inputtab[icase,0]    
    Uinf = Minf*cp.sqrt(1.4*rbar*Tinf)
    
    if Tinf > 170. :
        muinf = 17.e-6*(Tinf/273.1)**power
    else:
        muinf = 1.488e-6*cp.sqrt(Tinf)/(1. + (122.1/Tinf)*10.**(-5./Tinf))
    muw = muinf*(Twall/Tinf)**power
    rhow = rinf*(Tinf/Twall)
        
    # VD II transformation
    F = Twall/Tinf
    m = 0.2*Minf**2.
    A = cp.sqrt(recov*m/F)
    B = (1.+recov*m-F)/F
    alpha = (2.*A*A-B)/cp.sqrt(4.0*A*A+B*B)
    beta = B/cp.sqrt(4.*A*A+B*B)
    Fc = recov*m/(cp.arcsin(alpha)+cp.arcsin(beta))**2.
    Fth = muinf/muw
    Fx = (muinf/muw)*(cp.arcsin(alpha)+cp.arcsin(beta))**2.0/(recov*m)

    Reth0 = rinf*Uinf*1e-4/muinf
    Rethbar0 = Fth * Reth0
    Cfbar = 1./(17.08*cp.log10(Rethbar0)**2. + 25.11*cp.log10(Rethbar0) + 6.012)
    Cfbar0 = Cfbar
    Cf0  = Cfbar0/Fc
    def SivellsPayne(logRexbar):
        return 0.088*(logRexbar-2.3686)/(logRexbar-1.5)**3.0-cp.asnumpy(Cfbar)
    
    logRexbar = cp.asarray(fsolve(SivellsPayne,5))
    Rex0 = 10**logRexbar/Fx
    x0 = Rex0 * muinf/(rinf*Uinf)
    
    dReth = 1.
    Reth = Reth0
    for i in range(0,n):
        Reth = Reth+10**(float((i+1)*0.0015))
        Rethbar = Fth * Reth
        Cfbar = 1./(17.08*cp.log10(Rethbar)**2. + 25.11*cp.log10(Rethbar) + 6.012)
        Cf  = Cfbar/Fc
        logRexbar = cp.asarray(fsolve(SivellsPayne,5))
        Rex = 10**logRexbar/Fx
        x = Rex * muinf/(rinf*Uinf)
        xshift = x[0] - x0[0]
        utau = cp.sqrt(Cf*F/2)*Uinf
        delta = 0.37*x[0]/Rex[0]**0.2
        ztau = muw/rhow/utau
        Retau = rhow*utau*delta/muw
        if Retau > 510. :
            break
    print(Uinf, utau, ztau, Twall, Tinf, Minf, rinf, rhow, delta, Retau, x[0], xshift, Rex[0], file = f)

    
