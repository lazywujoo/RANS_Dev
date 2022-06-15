import cupy as cp
# from mesh import Mesh
from mesh2 import Mesh
from Global import input
from initialization import initfield
import GovEq
p=input()
# mesh=Mesh(p.nz,p.chi)
mesh = Mesh(p.nz, 8.70379880e-02, 6, 1)
inivar=initfield(p.nz, p.ztau, p.utau, p.Uinf,p.Tinf,p.gam,p.Minf,p.Tw,mesh.z)
res = 1.0e15
itcnt = 0
u = inivar.u
T = inivar.T
rho = inivar.rho
mu = inivar.mu
mut = inivar.mut
k = inivar.k
om = inivar.om
while res > 1.0e-6 and itcnt<1e6 :
    k,om,mut = GovEq.SST(u,k,om,mu,mut,rho,mesh,p)
    unm=u.copy()
    u = GovEq.Momentum(u,mut,mu,p,mesh,rho)
    rho,T,mu = GovEq.Algebraic(u,p,T,rho,mu)
    res = cp.linalg.norm(u-unm)/p.nz

    if itcnt%1 == 0: 
        print("iteration: ",itcnt, ", Residual(u) = ", res)
        print('--------u--------')
        print(u)    
    itcnt = itcnt + 1
