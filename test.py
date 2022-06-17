import cupy as cp
from mesh import Mesh
import time
# from mesh2 import Mesh
from Global import input
from initialization import initfield
import GovEq
import pdb
    
p=input()
mesh=Mesh(p.nz,p.chi)
# mesh = Mesh(p.nz, 8.70379880e-02, 6, 1)
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
start = time.time()
while res > 1.0e-3 and itcnt<1 :
    start2 = time.time()
    k,om,mut = GovEq.SST(u,k,om,mu,mut,rho,mesh,p)
    end2 = time.time()
    print('SST module time = ', end2 - start2)
    
    unm=u.copy()
    start2 = time.time()
    u = GovEq.Momentum(u,mut,mu,p,mesh,rho)
    end2 = time.time()
    print('momentum module time', end2 - start2)
    rho,T,mu = GovEq.Algebraic(u,p,T,rho,mu)
    start2 = time.time()
    print('algebraic module time = ', start2-end2)
    res = cp.linalg.norm(u-unm)/p.nz
    if itcnt%100 == 0: 
        print("iteration: ",itcnt, ", Residual(u) = ", res)
        # print('--------k--------')
        # print(k) 
        # print('--------u--------')
        # print(u)            
    itcnt = itcnt + 1
end = time.time()
print('total iteration time= ',end - start)


    