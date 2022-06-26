import matplotlib.pyplot as plt
import numpy as np
import pdb
from math import trunc
for n in range(0,1):
    tmp = trunc((3.0+n*0.1)*10)/10
    inputfile= "SolutionData/Solution"+str(n+1)+"_Minf" + str(tmp) 
    inputtable = np.loadtxt(inputfile,skiprows=3)
    zpls = inputtable[:,1]
    uplsvis = zpls
    uplslog = 1/0.41*np.log(zpls)+4.9
    uplslog[0] = 0
    uVD  = inputtable[:,3] 
    coff=np.where(zpls >30)
    icoff=coff[0]
    icoff=icoff[0]
    coff2=np.where(zpls >5)
    icoff2=coff2[0]
    icoff2=icoff2[0]
    plt.semilogx(zpls[1:-1],uVD[1:-1])
    plt.semilogx(zpls[1:icoff],uplsvis[1:icoff],'g--')
    plt.semilogx(zpls[icoff2:-1],uplslog[icoff2:-1],'g--')
    plt.xlabel('$z^+$', fontsize=16); plt.ylabel('$u^{vD}$', fontsize=16);
plt.savefig('test.png')    