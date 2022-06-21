def SST(u,k,om,mu,mut,rho,mesh,p):
    import numpy as np
    import cupy as cp
    import cupyx.scipy.sparse as cpx
    import tmp
    from cupyx.scipy.sparse import csr_matrix
    from cupyx.scipy.sparse.linalg import gmres
    import pdb
    # import pdb
    sigk1 = 0.85
    sigk2 = 1.0
    sigw1 = 0.5
    sigw2 = 0.856
    B1 = 0.075
    B2 = 0.0828
    Bstr = 0.09
    kap = 0.41 
    a1 = 0.31
    gam1 = B1/Bstr - sigw1*kap**2.0/cp.sqrt(Bstr)
    gam2 = B2/Bstr - sigw2*kap**2.0/cp.sqrt(Bstr) 
    dkdz  = mesh.ddz@k
    domdz = mesh.ddz@om
    
    underrelaxOm = 0.4
    underrelaxK = 0.6
    # blending functions
    CD_kom = cp.maximum(2.0*rho*sigw2/om*dkdz*domdz,1E-20)
    tmpvar1 = cp.sqrt(k)/(Bstr*om*mesh.z)
    tmpvar2 = 500.*mu/(rho*cp.power(mesh.z,2)*om)
    tmpvar3 = 4.*rho*sigw2*k/(CD_kom*cp.power(mesh.z,2))
    arg1 = cp.minimum(cp.maximum(tmpvar1,tmpvar2),tmpvar3)
    F1 = cp.tanh(cp.power(arg1,4))
    arg2 = cp.maximum(2*cp.sqrt(k)/(Bstr*om*mesh.z),500.*mu/(rho*cp.power(mesh.z,2)*om))
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


    #  -----------om ----------------------
    mueff = (mu + sigw*mut)/cp.sqrt(rho)
    dmueffdz = mesh.ddz*mueff
    # A  = mesh.d2dz2.multiply(mueff) + mesh.ddz.multiply(dmueffdz)
    A = cp.einsum('i,ij->ij', mueff, mesh.d2dz2.toarray()) \
        + cp.einsum('i,ij->ij',dmueffdz,mesh.ddz.toarray())
    # A  = tmparray#cp.asnumpy(mueff)
    # A2 = cp.zeros((p.nz,p.nz))
    # tmp = mesh.d2dz2.copy()
    # for m in range(0,p.nz):
    #     # A2[m,:] = mesh.d2dz2[m,:]*mueff[m] + mesh.ddz[m,:]*dmueffdz[m]
    #     A2[m,:] = tmp[m,:].toarray()*mueff[m]
    # print(A-A2)
    # pdb.set_trace()    
    # implicitly treated source term(diagonal) 
    # A = A - B_sp@(rho_sp.sqrt()@om_sp)
    cp.fill_diagonal(A, A.diagonal() - B*cp.sqrt(rho)*om)
    # for m in range(0,p.nz):    
    #     A2[m,m]= A2[m,m] - B[m] * cp.sqrt(rho[m])*om[m]
    # print(A-A2)
    # pdb.set_trace()
    # A = csr_matrix(A)
    # pdb.set_trace()
    # Boundary Condition
    # pdb.set_trace()
    om[0] = 60*mu[0]/(rho[0]*B1*mesh.z[1]**2)
    om[p.nz-1] = 10*p.Uinf/p.L
    # b matrix
    b = -gam*(rho*(strte*strte)) - (1-F1)*CD_kom
    # b= b_sp.diagonal(0)
    
    # Jacobi Preconditioner
    # D = cpx.diags(A[1:p.nz-1, 1:p.nz-1].diagonal(),offsets=0)
    # pdb.set_trace()
    # Ax = b Solver
    om = tmp.solveEqn(om*cp.sqrt(rho), A, b[1:p.nz-1], underrelaxOm,p.igmres)/cp.sqrt(rho)
    
    om[1:-1] = cp.maximum(om[1:-1], 1.e-12)

#  ---------------------------k -------------------------------------
    mueff = (mu + sigk*mut)/cp.sqrt(rho)
    # # initialize A sparse matrix with the size of d2dz2 sparse matrix
    dmueffdz = mesh.ddz@mueff
    # mueff_sp = cpx.diags(mueff)
    # dmueffdz_sp = cpx.diags(dmueffdz)    
    # A = mesh.d2dz2.copy()
    A = cp.einsum('i,ij->ij', mueff/cp.sqrt(rho), mesh.d2dz2.toarray()) \
        + cp.einsum('i,ij->ij',dmueffdz/cp.sqrt(rho),mesh.ddz.toarray())
    # A = mesh.d2dz2@mueff_sp + mesh.ddz@dmueffdz_sp
    # for m in range(0,p.nz):
    #     A[m,:] = mesh.d2dz2[m,:]*mueff[m] + mesh.ddz[m,:]*dmueffdz[m]

    # implicitly treated source term(diagonal)
    # A = A - (Bstr*cpx.eye(p.nz))@om_sp
    # A = csr_matrix(A)
    # for m in range(0,p.nz):    
    #     A[m,m]= (A[m,m] - Bstr*om[m])
    cp.fill_diagonal(A, A.diagonal() - Bstr*om)
    
    # Boundary Condition
    small=1E-20
    k[0] = small
    k[p.nz-1] = small

    # B matrix
    Pk = cp.minimum(mut*strte*strte, 20*Bstr*k*rho*om)
    b = -Pk
    underrelaxK = 0.6

    # Ax=b Solver
    k = tmp.solveEqn(k*rho, A, b[1:p.nz-1], underrelaxK,p.igmres)/rho
    k[1:-1] = cp.maximum(k[1:-1], 1.e-12)

# # ------------------------------------------------------------
#     # ---------------------------------------------------------------------
#     # om-equation
    
#     # effective viscosity
#     mueff = (mu + sigw*mut)/cp.sqrt(rho)
#     fs    = cp.sqrt(rho)


#     # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
#     A = cp.einsum('i,ij->ij', mueff, mesh.d2dz2.toarray())  \
#         + cp.einsum('i,ij->ij', mesh.ddz@mueff, mesh.ddz.toarray())
    
#     # implicitly treated source term
#     np.fill_diagonal(A, A.diagonal() - B*rho*om/fs)

#     # Right-hand-side
#     b = -gam[1:-1]*rho[1:-1]*strte[1:-1]*strte[1:-1] - (1-F1[1:-1])*CD_kom[1:-1]
    
#     # Wall boundary conditions
#     om[0 ] = 60.0*mu[0 ]/B1/rho[0 ]/mesh.z[1 ]/mesh.z[1 ]
#     om[-1] = 10*p.Uinf/p.L

#     # Solve
#     om = tmp.solveEqn(om*fs, A, b, underrelaxOm,p.igmres)/fs
#     om[1:-1] = np.maximum(om[1:-1], 1.e-12)
    
#     # ---------------------------------------------------------------------
#     # k-equation    
    
#     # effective viscosity

#     mueff = (mu + sigk*mut)/np.sqrt(rho)
#     fs    = rho
#     fd    = cp.sqrt(rho)


#     # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
#     A = cp.einsum('i,ij->ij', mueff*fd, mesh.d2dz2.toarray()) \
#       + cp.einsum('i,ij->ij', (mesh.ddz@mueff)*fd, mesh.ddz.toarray())

#     # implicitly treated source term
#     cp.fill_diagonal(A, A.diagonal() - Bstr*rho*om/fs)

#     # Right-hand-side
#     Pk = cp.minimum(mut*strte*strte, 20*Bstr*k*rho*om)
#     b  = -Pk[1:-1]
    
#     # Wall boundary conditions
#     k[0] = k[-1] = 1E-30
    
#     # Solve
#     k = tmp.solveEqn(k*fs, A, b, underrelaxK,p.igmres)/fs
#     k[1:-1] = np.maximum(k[1:-1], 1.e-12)
    return k,om,mut

def Momentum(u,mut,mu,p,mesh,rho):
    import cupy as cp
    import cupyx.scipy.sparse as cpx
    from cupyx.scipy.sparse import csr_matrix
    import tmp
    import pdb
    underrelaxU = 0.5
    mueff = mu+mut
    dmueffdz = mesh.ddz@mueff
    # mueff_sp = cpx.diags(mueff)
    # dmueffdz_sp = cpx.diags(dmueffdz)    
    # A = mesh.d2dz2@mueff_sp + mesh.ddz@dmueffdz_sp    
    # for m in range(0,p.nz):
    #     A[m,:] = mesh.d2dz2[m,:]*mueff[m] + mesh.ddz[m,:]*dmueffdz[m]
    A = cp.einsum('i,ij->ij', mueff, mesh.d2dz2.toarray()) + cp.einsum('i,ij->ij',dmueffdz,mesh.ddz.toarray())


    b=cp.zeros((p.nz))
    b2=cp.zeros((p.nz))
    # # b matrix
    coff=cp.where(mesh.z/p.ztau >30)
    coff2 = cp.where(mesh.z/p.delta >0.4)
    icoff=coff[0]
    icoff2=coff2[0]
    tauw=-p.rhow* p.utau*p.utau 

    b[0:icoff[0]] = tauw/p.delta/float(icoff2[0]-icoff[0])
    b2[icoff2[0]:p.nz] = tauw/p.delta/float(icoff2[0]-icoff[0])
    # b_sp = tauw*cpx.eye(p.nz)-cpx.diags(b)-cpx.diags(b2)
    b= tauw*cp.ones(p.nz)/p.delta/float(icoff2[0]-icoff[0])-b-b2
    # pdb.set_trace()

    # pdb.set_trace()
    # for m in range(0,p.nz):
    #     if  (mesh.z[m]/p.ztau >400):
    #         b[m] = 0      
    #     else:       
    #         b[m] = -rho[0]* p.utau*p.utau 
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

    if p.itempeq == 1:
        Tr = p.Tinf*(1+0.9*(p.gam-1)/2*p.Minf**2)
        func_u = 0.1741*(u/p.Uinf)**2 + 0.8259*(u/p.Uinf)
        T   = p.Tinf*((p.Tw/p.Tinf)+(Tr-p.Tw)/p.Tinf*func_u+(p.Tinf-Tr)/p.Tinf*(u/p.Uinf)**2)

        rw = rho[0]
        rho = rw*p.Tw/T  

    # mu = 1.458E-6*cp.sqrt(p.Tinf**3/2 )/(p.Tinf+110.3)*(T/p.Tinf)**0.76   
    mu = 1.458E-6*cp.sqrt(T**3)/(T+110.3)
    return rho,T,mu

