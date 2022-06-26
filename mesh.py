import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
import math as m
# import pdb

class Mesh:
    def __init__(self, p):
        n = p.nz
        chi = p.chi
        ztmp = cp.zeros(n)
        ztmp[0] = 0
        ztmp[1] = p.zpls2*p.ztau
        kcnt=2
        for k in range(2,n):
            ztmp[k] = (ztmp[k-1]-ztmp[k-2])*chi+ztmp[k-1]
            # if ztmp[k] > p.delta:
            #     ztmp[k] = p.delta
            #     break
            kcnt=kcnt+1
        z = ztmp[0:kcnt]
        n = kcnt
        p.nz = kcnt
        self.z = z
        
        dchidz =np.zeros(n)
        # ddy=d/dchi*dchi/dz 
        # 4th order forward difference for dchi/dz at the first grid point
        # 4th order backward difference for dchi/dz at the last grid point
        dchidz[0] = 1/(-25/12*z[0]+4.*z[1]-3.*z[2]+4/3*z[3]-0.25*z[4])
        
        for k in range(1,n-1):
            dchidz[k] = 1./(-0.5*z[k-1]+0.5*z[k+1])
        nm = n -1 
        dchidz[nm] = 1./(-25/12*z[nm]+4.*z[nm-1]-3.*z[nm-2]+4/3*z[nm-3]-0.25*z[nm-4]) 
        # converting numpy array to cupy array
        dchidz=cp.array(dchidz)
        

        ddchi=np.zeros((n,n))
        ddchi[0,0:5] = [-25/12, 4. ,-3. ,4/3 ,-0.25] 
        for k in range(1,nm):
            ddchi[k,k-1:k+2] = [-0.5, 0. , 0.5]
        ddchi[nm,nm-4:n] = [-0.25, 4/3, -3., 4., -25/12]

        # converting numpy array to cupy array
        ddchi=cp.array(ddchi)
        ddchi=csr_matrix(ddchi)
        # scalar multiplication
        ddz=ddchi
        for k in range(0,n):
            ddz[k,:] = ddchi[k,:]*dchidz[k]
        self.ddz=ddz
        # !!!!!!!!!!!!! double check below grid transformation with someone else !!!!!!!!!!
        # d2u/dz2=d2u/dchi2(dchi/dz)^2 - du/dchi*(d2zdchi2)(dchidz)^3
        # 4th order forward difference for ddu/dyy at the first grid point
        # 4th order backward difference for ddu/dyy at the last grid point
        d2zdchi2 =np.zeros(n)
        d2zdchi2[0] = (15/4*z[0]-77/6*z[1]+107/6*z[2]-13.*z[3]+61/12*z[4]-5/6*z[5])
        
        for k in range(1,n-1):
            d2zdchi2[k] = (z[k-1]-2.*z[k]+z[k+1])
        d2zdchi2[nm] = (15/4*z[nm]-77/6*z[nm-1]+107/6*z[nm-2]-13.*z[nm-3]+61/12*z[nm-4]-5/6*z[nm-5])
        # converting numpy array to cupy array
        d2zdchi2=cp.array(d2zdchi2)

        d2udchi2=np.zeros((n,n))
        d2udchi2[0,0:6] = [15/4, -77/6, 107/6, -13., 61/12,-5/6] 
        for k in range(1,nm):
            d2udchi2[k,k-1:k+2] = [1., -2. , 1.]
        d2udchi2[nm,nm-5:n] = [-5/6, 61/12, -13., 107/6, -77/6, 15/4]
        d2udchi2 = cp.array(d2udchi2)
        d2udchi2 = csr_matrix(d2udchi2)

        d2dz2 = d2udchi2
        for k in range(0,n):
            d2dz2[k,:]= d2udchi2[k,:]*cp.power(dchidz[k],2)-ddz[k,:]*d2zdchi2[k]*cp.power(dchidz[k],2)
        self.d2dz2 = d2dz2
