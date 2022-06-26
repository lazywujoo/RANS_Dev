import cupy as cp
from mesh import Mesh
import time
# from mesh2 import Mesh
from Global import input
from initialization import initfield
import GovEq
import pdb

inputfile = "1D-HBL.inp"
inputtable = cp.loadtxt(inputfile,skiprows=1) 

# for n in range(0,inputtable.shape[0]):
for n in range(0,1):
    print('running Minf = ', inputtable[n,5])     
    p=input(inputtable[n,:])
    mesh=Mesh(p)
    print(p.nz)
    inivar=initfield(p.nz, p.ztau, p.utau, p.Uinf,p.Tinf,p.gam,p.Minf,p.Tw,mesh.z,p.rhow)
    res = 1.0e15
    itcnt = 0
    u = inivar.u
    T = inivar.T
    rho = inivar.rho
    mu = inivar.mu
    mut = inivar.mut
    k = inivar.k
    om = inivar.om
    v2 = inivar.v2
    e = inivar.e
    if p.itempeq == 0:
        file = "DNS_Data/M2T5_Stat.dat"
        DNS = cp.loadtxt(file,skiprows=1)
    # interpolate density from DNS on RANS mesh
        rho = cp.interp(mesh.z, DNS[:,0], DNS[:,7])*p.rhow
        T   = cp.interp(mesh.z, DNS[:,0], DNS[:,6])*p.Tinf
    start = time.time()
    while res > 1.0e-4 and itcnt<1e6:
        if p.iRANS == 1:
            k,om,mut = GovEq.SST(u,k,om,mu,mut,rho,mesh,p)
        elif p.iRANS == 2:
            tauw=mu[0]*(u[1]-u[0])/(mesh.z[1]-mesh.z[0])
            utau = cp.sqrt(tauw/p.rhow)
            mut = GovEq.Cess(rho,mu,p,mesh,p.utau)
        elif p.iRANS == 3:
            mut,k,e,v2 = GovEq.V2F(u,k,e,v2,rho,mu,mesh,p)
        # pdb.set_trace()
        unm=u.copy()
        u = GovEq.Momentum(u,mut,mu,p,mesh,rho)
        rho,T,mu = GovEq.Algebraic(u,p,T,rho,mu)
        res = cp.linalg.norm(u-unm)/p.nz
        if itcnt%100 == 0:  
            print("iteration: ",itcnt, ", Residual(u) = ", res)    
        itcnt = itcnt + 1
        # pdb.set_trace()        
    end = time.time()
    tauw=mu[0]*(u[1]-u[0])/(mesh.z[1]-mesh.z[0])
    utau = cp.sqrt(tauw/p.rhow)
    ztau = mu[0]/p.rhow/utau   
    print("iteration: ",itcnt, ", Residual(u) = ", res)
    print('total iteration time= ',end - start)
    print('calculated utau =', utau, 'input utau = ', p.utau)

    def VDT(u,rho,p,mesh,utau,ztau):
        u=u/utau
        uvd = cp.zeros(p.nz)
        zpls = mesh.z/ztau
        
        for m in range(1,p.nz):
            uvd[m] = uvd[m-1] + cp.sqrt(0.5*(rho[m]+rho[m-1])/p.rhow)*(u[m]-u[m-1])
            
        return zpls,uvd
    
    zpls,uVD = VDT(u,rho,p,mesh,utau,ztau)
    saveflnm = "SolutionData/Solution"+str(n+1)+"_Minf" + str(inputtable[n,5]) 
    f= open(saveflnm, "w")
    f.write('Uinf utau ztau Twall Tinf Minf rhoinf rhow delta \n')
    print(p.Uinf, utau, ztau, p.Tw, p.Tinf, p.Minf, p.rinf, p.rhow, p.delta, file=f)
    f.write("variables= z zpls u upls uVD rho rhopls T Tpls k om mu mut \n")
    for m in range(0,p.nz):
        print(mesh.z[m], zpls[m], u[m], u[m]/p.utau , uVD[m], rho[m] ,rho[m]/rho[0], T[m] ,  \
            T[m]/p.Tw, k[m], om[m], mu[m], mut[m], file=f)
    f.close                  



    