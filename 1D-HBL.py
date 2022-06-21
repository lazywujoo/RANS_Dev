import cupy as cp
from mesh import Mesh
import time
# from mesh2 import Mesh
from Global import input
from initialization import initfield
import GovEq
import pdb
    
p=input()
mesh=Mesh(p)
# mesh = Mesh(p.nz, 8.70379880e-02, 6, 1)
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
if p.itempeq == 0:
    file = "DNS_Data/M2T5_Stat.dat"
    DNS = cp.loadtxt(file,skiprows=1)
# interpolate density from DNS on RANS mesh
    rho = cp.interp(mesh.z, DNS[:,0], DNS[:,7])*p.rhow
    T   = cp.interp(mesh.z, DNS[:,0], DNS[:,6])*p.Tinf
start = time.time()
while res > 1.0e-4 and itcnt<1e6:
    k,om,mut = GovEq.SST(u,k,om,mu,mut,rho,mesh,p)
    
    unm=u.copy()
    u = GovEq.Momentum(u,mut,mu,p,mesh,rho)
    rho,T,mu = GovEq.Algebraic(u,p,T,rho,mu)
    res = cp.linalg.norm(u-unm)/p.nz
    if itcnt%100 == 0:  
        print("iteration: ",itcnt, ", Residual(u) = ", res)
        
    itcnt = itcnt + 1
    
end = time.time()
tauw=mu[0]*(u[1]-u[0])/(mesh.z[1]-mesh.z[0])
utau=cp.sqrt(tauw/p.rhow)
print("iteration: ",itcnt, ", Residual(u) = ", res)
print('total iteration time= ',end - start)
print('calculated utau =', utau, 'input utau = ', p.utau)

def VDT(u,rho,p,mesh,utau):
    u=u/utau
    uvd = cp.zeros(p.nz)
    zpls = mesh.z/p.ztau
    
    for m in range(1,p.nz):
        uvd[m] = uvd[m-1] + cp.sqrt(0.5*(rho[m]+rho[m-1])/p.rhow)*(u[m]-u[m-1])
        
    return zpls,uvd

zpls,uVD = VDT(u,rho,p,mesh,utau)
f= open("Solution.dat", "w")
f.write('Uinf utau ztau Twall Tinf Minf rhoinf rhow delta \n')
print(p.Uinf, utau, p.ztau, p.Tw, p.Tinf, p.Minf, p.rinf, p.rhow, p.delta, file=f)
f.write("variables= z zpls upls uVD rho rhopls T Tpls k om mu mut \n")
for m in range(0,p.nz):
    print(mesh.z[m], zpls[m], u[m]/p.utau , uVD[m], rho[m] ,rho[m]/rho[0], T[m] ,  \
        T[m]/p.Tw, k[m], om[m], mu[m], mut[m], file=f)



    