import pdb


def SST(u,k,om,mu,mut,rho,mesh,p):
    import pdb
    import cupy as cp
    import cupyx.scipy.sparse as cpx
    import tmp
    from cupyx.scipy.sparse import csr_matrix
    from cupyx.scipy.sparse.linalg import gmres

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
    # pdb.set_trace()
    #  -----------om ----------------------
    mueff = (mu + sigw*mut)/cp.sqrt(rho)
    A=mesh.d2dz2.copy()
    dmueffdz = mesh.ddz@mueff
    for m in range(0,p.nz):
        A[m,:] = mesh.d2dz2[m,:]*mueff[m] + mesh.ddz[m,:]*dmueffdz[m]

    # implicitly treated source term(diagonal) 
    for m in range(0,p.nz):    
        A[m,m]= A[m,m] - B[m] * cp.sqrt(rho[m])*om[m]

    # Boundary Condition
    om[0] = 60*mu[0]/(rho[0]*B1*mesh.z[1]**2)
    om[p.nz-1] = 3*p.Uinf/p.L
    # pdb.set_trace()
    # B matrix
    b = -gam*rho*strte*strte - (1-F1)*CD_kom
    # tmp2 = om[0]*A[1:p.nz-1,0]
    # b[1] = b[1] - tmp2[0,0]
    # tmp2 = om[p.nz-1]*A[1:p.nz-1,p.nz-1] 
    # b[p.nz-2] = b[p.nz-2]- tmp2[p.nz-3,0]
    # pdb.set_trace()
    # Jacobi Preconditioner
    D = cpx.diags(A[1:p.nz-1, 1:p.nz-1].diagonal(),offsets=0)

    underrelaxOm = 0.4

    # Solving
    # sol = gmres(A[1:p.nz-1, 1:p.nz-1],b[1:p.nz-1],cp.sqrt(rho[1:p.nz-1])*om[1:p.nz-1], tol=1e-6, maxiter=1 )
    # om[1:p.nz-1]=sol[0]/cp.sqrt(rho[1:p.nz-1])
    om = tmp.solveEqn(om*cp.sqrt(rho), A, b[1:p.nz-1], underrelaxOm)/cp.sqrt(rho)

    om[1:-1] = cp.maximum(om[1:-1], 1.e-12)

#  ---------------------------k -------------------------------------
    mueff = (mu + sigk*mut)/cp.sqrt(rho)
    # # initialize A sparse matrix with the size of d2dz2 sparse matrix
    dmueffdz = mesh.ddz@mueff
    A = mesh.d2dz2.copy()
    for m in range(0,p.nz):
        A[m,:] = mesh.d2dz2[m,:]*mueff[m] + mesh.ddz[m,:]*dmueffdz[m]

    # implicitly treated source term(diagonal)
    for m in range(0,p.nz):    
        A[m,m]= (A[m,m] - Bstr*om[m])
    # cp.fill_diagonal(A, A.diagonal() - Bstr*om)
    
    # Boundary Condition
    small=1E-20
    k[0] = small
    k[p.nz-1] = small

    # B matrix
    Pk = cp.minimum(mut*strte*strte, 20*Bstr*k*rho*om)
    b = -Pk

    # tmp2 = k[0]*A[1:p.nz-1,0]
    # b[1] = b[1] - tmp2[0,0]
    # tmp2 = k[p.nz-1]*A[1:p.nz-1,p.nz-1] 
    # b[p.nz-2] = b[p.nz-2]- tmp2[p.nz-3,0]

    # Jacobi Preconditioner
    D = cpx.diags(A[1:p.nz-1, 1:p.nz-1].diagonal(),offsets=0)
    underrelaxK = 0.6
    # pdb.set_trace()
    # sol = gmres(A[1:p.nz-1, 1:p.nz-1],b[1:p.nz-1],rho[1:p.nz-1]*k[1:p.nz-1],tol=1e-6, maxiter=1)
    # k[1:p.nz-1]=sol[0]/(rho[1:p.nz-1])
    k = tmp.solveEqn(k*rho, A, b[1:p.nz-1], underrelaxK)/rho
    k[1:-1] = cp.maximum(k[1:-1], 1.e-12)


    return k,om,mut

def Momentum(u,mut,mu,p,mesh,rho):
    # import pdb
    import cupy as cp
    import cupyx.scipy.sparse.linalg as cpx
    import tmp
    mueff = mu+mut

    A=mesh.d2dz2.copy()
    dmueffdz = mesh.ddz@mueff
    for m in range(0,p.nz):
        A[m,:] = mesh.d2dz2[m,:]*mueff[m] + mesh.ddz[m,:]*dmueffdz[m]
    # pdb.set_trace( )
    b=cp.zeros((p.nz))
    for m in range(0,p.nz):
        if  (mesh.z[m]/p.ztau >400):
            b[m] = 0      
        else:       
            b[m] = -rho[0]* p.utau*p.utau 
    u[0] = 0
    u[p.nz-1] = p.Uinf
    # tmp2 = u[p.nz-1] * A[1:p.nz-1,p.nz-1]
    
    # b[p.nz-2] = b[p.nz-2] - tmp2[p.nz-3,0]

    # sol = cpx.gmres(A[1:p.nz-1,1:p.nz-1],b[1:p.nz-1],u[1:p.nz-1],tol=1e-6, maxiter=1)
    # u[1:p.nz-1] = sol[0]
    # pdb.set_trace( )
    # u[1:p.nz-1] = cpx.spsolve(A[1:p.nz-1, 1:p.nz-1], b[1:p.nz-1])
    # pdb.set_trace( )
    underrelaxU = 0.5
    u = tmp.solveEqn(u, A, b[1:p.nz-1], underrelaxU)
    u[1:-1] = cp.maximum(u[1:-1], 0.0)

    return u

def Algebraic(u,p,T,rho,mu):
    import cupy as cp
    Tr = p.Tinf*(1+0.9*(p.gam-1)/2*p.Minf**2)
    T[0] = p.Tw
    for m in range(1,p.nz):
        func_u = 0.1741*(u[m]/p.Uinf)**2 + 0.8259*(u[m]/p.Uinf)
        T[m]   = p.Tinf*(p.Tw/p.Tinf+(Tr-p.Tw)/p.Tinf*func_u+(p.Tinf-Tr)/p.Tinf*(u[m] /p.Uinf)**2)

    for m in range(1,p.nz):
        rho[m] = rho[0]*p.Tw/T[m]   

    for m in range(0,p.nz):
        mu[m] = 1.458E-6*cp.sqrt(p.Tinf**3)/(p.Tinf+110.3)*(T[m]/p.Tinf)**0.76     

    return rho,T,mu
