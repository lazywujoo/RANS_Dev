import pdb


def SST(u,k,om,mu,mut,rho,mesh,p):
    import pdb
    import time
    import cupy as cp
    import cupyx.scipy.sparse as cpx
    import tmp
    from cupyx.scipy.sparse import csr_matrix
    from cupyx.scipy.sparse.linalg import gmres

    start = time.time()
    sigk1 = 0.85
    sigk2 = 1.0
    sigw1 = 0.5
    sigw2 = 0.856
    B1 = 0.075
    B2 = 0.0828
    Bstr = 0.09
    kap = 0.41 
    a1 = 0.31
    gam1 = B1/Bstr - sigw1*kap**2/cp.sqrt(Bstr)
    gam2 = B2/Bstr - sigw2*kap**2/cp.sqrt(Bstr) 
    dkdz  = mesh.ddz@k
    domdz = mesh.ddz@om
    
    underrelaxOm = 0.4
    # blending functions
    CD_kom = cp.maximum(2*rho*sigw2/om*dkdz*domdz,1E-20)
    tmpvar1 = cp.sqrt(k)/(Bstr*om*mesh.z)
    tmpvar2 = 500*mu/(rho*cp.power(mesh.z,2)*om)
    tmpvar3 = 4*rho*sigw2*k/(CD_kom*cp.power(mesh.z,2))
    arg1 = cp.minimum(cp.maximum(tmpvar1,tmpvar2),tmpvar3)
    F1 = cp.tanh(cp.power(arg1,4))
    arg2 = cp.maximum(2*cp.sqrt(k)/(Bstr*om*mesh.z),500*mu/(rho*cp.power(mesh.z,2)*om))
    F2 = cp.tanh(cp.power(arg2,2))

    strte= cp.absolute(mesh.ddz@u)
    #  turbulent viscosity
    zeta = cp.minimum(1.0/om, a1/(strte*F2))
    mut = rho*k*zeta
    # do I need this limiter for turb_vis?? from pesnik
    mut = cp.minimum(cp.maximum(mut,0.0),100.0)

    # blending function adjusted constants
    sigk = sigk1*F1+(1-F1)*(sigk2)
    sigw = sigw1*F1+(1-F1)*(sigw2)
    B = B1*F1+(1-F1)*(B2)
    gam = gam1*F1+(1-F1)*(gam2)

    B_sp = cpx.diags(B)
    rho_sp = cpx.diags(rho)
    om_sp = cpx.diags(om)
    strte_sp = cpx.diags(strte)
    F1_sp = cpx.diags(F1)
    CD_kom_sp = cpx.diags(CD_kom)
    gam_sp = cpx.diags(-gam)

    end= time.time()
    print('SST var init time = ', end-start)


    #  -----------om ----------------------
    start = time.time()
    mueff = (mu + sigw*mut)/cp.sqrt(rho)
    A=mesh.d2dz2.copy()
    dmueffdz = mesh.ddz@mueff
    mueff_sp = cpx.diags(mueff)
    dmueffdz_sp = cpx.diags(dmueffdz)
    A = mesh.d2dz2@mueff_sp + mesh.ddz@dmueffdz_sp
    # for m in range(0,p.nz):
    #     A[m,:] = mesh.d2dz2[m,:]*mueff[m] + mesh.ddz[m,:]*dmueffdz[m]
    end = time.time()   
    print('A(om) matrix init time = ', end-start)
    
    start = time.time()
    # implicitly treated source term(diagonal) 
    A = A - B_sp@(rho_sp.sqrt()@om_sp)
    # for m in range(0,p.nz):    
    #     A[m,m]= A[m,m] - B[m] * cp.sqrt(rho[m])*om[m]
    end = time.time()
    print('implicit treatment of A = ', end-start)
    
    # Boundary Condition
    om[0] = 60*mu[0]/(rho[0]*B1*mesh.z[1]**2)
    om[p.nz-1] = 3*p.Uinf/p.L

    # b matrix
    start = time.time()
    b_sp = gam_sp@(rho_sp@(strte_sp@strte_sp)) - (cpx.eye(p.nz)-F1_sp)@CD_kom_sp
    b= b_sp.diagonal(0)
    end = time.time()
    print('b matrix time = ', end-start)
    
    # Jacobi Preconditioner
    # D = cpx.diags(A[1:p.nz-1, 1:p.nz-1].diagonal(),offsets=0)
    
    # Ax = b Solver
    start = time.time()
    om = tmp.solveEqn(om*cp.sqrt(rho), A, b[1:p.nz-1], underrelaxOm,p.igmres)/cp.sqrt(rho)
    end = time.time()
    print('om solver time',end-start)
    
    om[1:-1] = cp.maximum(om[1:-1], 1.e-12)

#  ---------------------------k -------------------------------------
    mueff = (mu + sigk*mut)/cp.sqrt(rho)
    # # initialize A sparse matrix with the size of d2dz2 sparse matrix
    dmueffdz = mesh.ddz@mueff
    mueff_sp = cpx.diags(mueff)
    dmueffdz_sp = cpx.diags(dmueffdz)    
    # A = mesh.d2dz2.copy()
    A = mesh.d2dz2@mueff_sp + mesh.ddz@dmueffdz_sp
    # for m in range(0,p.nz):
    #     A[m,:] = mesh.d2dz2[m,:]*mueff[m] + mesh.ddz[m,:]*dmueffdz[m]

    # implicitly treated source term(diagonal)
    A = A - (Bstr*cpx.eye(p.nz))@om_sp
    # for m in range(0,p.nz):    
    #     A[m,m]= (A[m,m] - Bstr*om[m])
    # cp.fill_diagonal(A, A.diagonal() - Bstr*om)
    
    # Boundary Condition
    small=1E-20
    k[0] = small
    k[p.nz-1] = small

    # B matrix
    start = time.time()
    Pk = cp.minimum(mut*strte*strte, 20*Bstr*k*rho*om)
    b = -Pk
    end = time.time()
    print('b matrix for k', end-start)
    # Jacobi Preconditioner
    D = cpx.diags(A[1:p.nz-1, 1:p.nz-1].diagonal(),offsets=0)
    underrelaxK = 0.6

    # Ax=b Solver
    k = tmp.solveEqn(k*rho, A, b[1:p.nz-1], underrelaxK,p.igmres)/rho
    k[1:-1] = cp.maximum(k[1:-1], 1.e-12)


    return k,om,mut

def Momentum(u,mut,mu,p,mesh,rho):
    import cupy as cp
    import cupyx.scipy.sparse as cpx
    import tmp
    import time 
    import pdb

    underrelaxU = 0.5
    mueff = mu+mut
    dmueffdz = mesh.ddz@mueff
    mueff_sp = cpx.diags(mueff)
    dmueffdz_sp = cpx.diags(dmueffdz)    
    A = mesh.d2dz2@mueff_sp + mesh.ddz@dmueffdz_sp    
    # for m in range(0,p.nz):
    #     A[m,:] = mesh.d2dz2[m,:]*mueff[m] + mesh.ddz[m,:]*dmueffdz[m]
    
    b=cp.zeros((p.nz))
    # b matrix
    start = time.time()
    coff=cp.where(mesh.z/p.ztau >400)
    
    icoff=coff[0]
    tauw=-rho[0]* p.utau*p.utau 
    b[icoff[0]:p.nz] = tauw
    b_sp = tauw*cpx.eye(p.nz)-cpx.diags(b)
    b= b_sp.diagonal(0)
    # for m in range(0,p.nz):
    #     if  (mesh.z[m]/p.ztau >400):
    #         b[m] = 0      
    #     else:       
    #         b[m] = -rho[0]* p.utau*p.utau 
    end = time.time()
    print('momentum b matrix time = ', end-start)
    # Boundary Condition
    u[0] = 0
    u[p.nz-1] = p.Uinf

    # Ax = b Solver
    u = tmp.solveEqn(u, A, b[1:p.nz-1], underrelaxU,p.igmres)
    u[1:-1] = cp.maximum(u[1:-1], 0.0)

    return u


def Algebraic(u,p,T,rho,mu):
    import cupy as cp
    import cupyx.scipy.sparse as cpx
    # from numba import cuda,vectorize
    import time
    
    # @vectorize(['float64(float64, float64,float64, float64,float64, float64)'], target='cuda')
    # def Temperature(u,Tinf,gam,Minf,Tw,Uinf):
    #     Tr = Tinf*(1+0.9*(gam-1)/2*Minf**2)
        
    #     func_u = 0.1741*(u/Uinf)**2 + 0.8259*(u/Uinf)
    #     T   = Tinf*(Tw/Tinf+(Tr-Tw)/Tinf*func_u+(Tinf-Tr)/Tinf*(u /Uinf)**2)
    #     return T
    # start=time.time()
    # Tinf = p.Tinf*cp.ones(p.nz,dtype = float)
    # gam = p.gam*cp.ones(p.nz,dtype = float)
    # Minf = p.Minf*cp.ones(p.nz,dtype = float)
    # Tw = p.Tw*cp.ones(p.nz,dtype = float)
    # Uinf = p.Uinf*cp.ones(p.nz,dtype = float)

    # T = Temperature(u,Tinf,gam,Minf,Tw,Uinf)
    # T[0] = p.Tw
    # end = time.time()
    # print('temperature algebraic time =', end-start)

    Tr = p.Tinf*(1+0.9*(p.gam-1)/2*p.Minf**2)
    Tr_sp = Tr*cpx.eye(p.nz)
    Tinf_sp = p.Tinf*cpx.eye(p.nz)
    Uinf_sp = p.Uinf*cpx.eye(p.nz)
    u_sp = cpx.diags(u)    
    Tw_sp = p.Tw*cpx.eye(p.nz)
    func_u = 0.1741*(u/p.Uinf)**2 + 0.8259*(u/p.Uinf)
    func_u = cpx.diags(func_u)
    T   = p.Tinf*((p.Tw/p.Tinf*cpx.eye(p.nz))+(Tr-p.Tw)/p.Tinf*func_u+(p.Tinf-Tr)/p.Tinf*(u_sp/p.Uinf)**2)

    rw = rho[0]
    rho = rw*p.Tw/T.diagonal()   

    mu = 1.458E-6*cp.sqrt(p.Tinf**3)/(p.Tinf+110.3)*(T.diagonal()/p.Tinf)**0.76     
    T = T.diagonal()
    return rho,T,mu

