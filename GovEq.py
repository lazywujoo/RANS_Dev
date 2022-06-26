def SST(u,k,om,mu,mut,rho,mesh,p):
    import numpy as np
    import cupy as cp
    import cupyx.scipy.sparse as cpx
    import Solver
    from cupyx.scipy.sparse import csr_matrix
    from cupyx.scipy.sparse.linalg import gmres
    # import pdb
    import pdb
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

    # pdb.set_trace()
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
    om[0] = 60*mu[-1]/(rho[-1]*B1*mesh.z[1]**2)
    om[p.nz-1] = 10*p.Uinf/p.L
    # b matrix
    b = -gam*(rho*(strte*strte)) - (1-F1)*CD_kom
    # b= b_sp.diagonal(0)
    
    # Jacobi Preconditioner
    # D = cpx.diags(A[1:p.nz-1, 1:p.nz-1].diagonal(),offsets=0)
    # pdb.set_trace()
    # Ax = b Solver
    om = Solver.solveEqn(om*cp.sqrt(rho), A, b[1:p.nz-1], underrelaxOm,p.igmres)/cp.sqrt(rho)
    
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

    cp.fill_diagonal(A, A.diagonal() - Bstr*om)
    
    # Boundary Condition
    small=1E-20
    k[0] = small
    k[p.nz-1] = 0.02

    # B matrix
    Pk = cp.minimum(mut*strte*strte, 20*Bstr*k*rho*om)
    b = -Pk
    underrelaxK = 0.6

    # Ax=b Solver
    k = Solver.solveEqn(k*rho, A, b[1:p.nz-1], underrelaxK,p.igmres)/rho
    k[1:-1] = cp.maximum(k[1:-1], 1.e-12)

    return k,om,mut

def Momentum(u,mut,mu,p,mesh,rho):
    import cupy as cp
    import cupyx.scipy.sparse as cpx
    from cupyx.scipy.sparse import csr_matrix
    import Solver
    import pdb
    underrelaxU = 0.5
    mueff = mu + mut
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
    coff2 = cp.where(mesh.z/p.delta >1.0)
    icoff=coff[0]
    icoff2=coff2[0]
    tauw=-p.rhow* p.utau*p.utau/p.delta

    b[0:icoff[0]] = tauw
    b2[icoff2[0]:p.nz] = tauw
    b= tauw*cp.ones(p.nz)-b2
    # Boundary Condition
    u[0] = 0
    u[p.nz-1] = p.Uinf
    # Ax = b Solver
    u = Solver.solveEqn(u, A, b[1:p.nz-1], underrelaxU,0)
    # u = Solver.solveEqn2(u, A, b[1:p.nz-1], underrelaxU,0,mesh)
    
    u[1:-1] = cp.maximum(u[1:-1], 0.0)
    u[1:-1] = cp.minimum(u[1:-1], p.Uinf)

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

def Cess(r,mu,p,mesh,utau):
    import cupy as cp 
    ReTau = p.rhow*utau*p.delta/mu[0]
    # Model constants
    kappa   = 0.426
    A       = 25.4
    ReTauArr = cp.sqrt(r/r[0])/(mu/mu[0])*ReTau
    zplus = mesh.z*ReTauArr
     
    df  = 1 - cp.exp(-zplus/A)
    t1  = cp.power(2*mesh.z-mesh.z*mesh.z, 2)
    t2  = cp.power(3-4*mesh.z+2*mesh.z*mesh.z, 2)
    mut = 0.5*cp.power(1 + 1/9*cp.power(kappa*ReTauArr, 2)*(t1*t2)*df*df, 0.5) - 0.5
    
    return mut*mu

def V2F(u,k,e,v2,r,mu,mesh,p):
    import cupy as cp
    import cupyx.scipy.sparse as cpx
    import Solver
    from cupyx.scipy.sparse import csr_matrix
    from cupyx.scipy.sparse.linalg import gmres
    import pdb
    small = 1e-20
    n = p.nz
    f = cp.zeros(n)

    # Model constants
    cmu  = 0.22 
    sigk = 1.0 
    sige = 1.3 
    Ce2  = 1.9
    Ct   = 6 
    Cl   = 0.23 
    Ceta = 70 
    C1   = 1.4 
    C2   = 0.3
    # Relaxation factors
    underrelaxK  = 0.8
    underrelaxE  = 0.8
    underrelaxV2 = 0.8
    # pdb.set_trace()

    # Time and length scales, eddy viscosity and turbulent production
    Tt  = cp.maximum(k/e, Ct*cp.power(mu/(r*e), 0.5))
    Lt  = Cl*cp.maximum(cp.power(k, 1.5)/e, Ceta*cp.power(cp.power(mu/r, 3)/e, 0.25))
    mut = cp.maximum(cmu*r*v2*Tt, 0.0)
    Pk  = mut*cp.power(cp.absolute(mesh.ddz@u), 2.0)


    # ---------------------------------------------------------------------
    # f-equation 
    
    # implicitly treated source term
    A = cp.einsum('i,ij->ij',Lt*Lt, mesh.d2dz2.toarray())
    cp.fill_diagonal(A, A.diagonal() - 1.0)
    
    # Right-hand-side
    vok  = v2/k
    rhsf = ((C1-6)*vok - 2/3*(C1-1))/Tt - C2*Pk/(r*k)
    
    # Solve
    f = Solver.solveEqn(f,A,rhsf[1:p.nz-1],underrelaxK,p.igmres)
    f[1:p.nz-1] = cp.maximum(f[1:p.nz-1], 1.e-12)

    
    # ---------------------------------------------------------------------
    # v2-equation: 
    
    # effective viscosity and pre-factors for compressibility implementation
    mueff = (mu + mut)/cp.sqrt(r)
    fs    = r
    fd    = 1/cp.sqrt(r)


    # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
    A = cp.einsum('i,ij->ij', mueff*fd, mesh.d2dz2.toarray()) \
      + cp.einsum('i,ij->ij', (mesh.ddz@mueff)*fd, mesh.ddz.toarray())

    # implicitly treated source term
    cp.fill_diagonal(A, A.diagonal() - 6.0*r*e/k/fs)
    
    # Right-hand-side
    b = -r*k*f
    
    # Wall boundary conditions
    v2[0]  = small

    # Solve
    v2 = Solver.solveEqn2(v2*fs,A,b[1:p.nz-1],underrelaxV2,p.igmres,mesh)/fs
    # v2 = Solver.solveEqn(v2*fs,A,b[1:p.nz-1],underrelaxV2,p.igmres)/fs
    v2[1:p.nz-1] = cp.maximum(v2[1:p.nz-1], 1.e-12)
    # pdb.set_trace()
    
    # ---------------------------------------------------------------------
    # e-equation
        
    # effective viscosity
    mueff = (mu + mut/sige)/cp.sqrt(r)
    fs    = cp.power(r, 1.5)
    fd    = 1/r

    # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
    A = cp.einsum('i,ij->ij', mueff*fd, mesh.d2dz2.toarray()) \
      + cp.einsum('i,ij->ij', (mesh.ddz@mueff)*fd, mesh.ddz.toarray())
    
    # implicitly treated source term
    cp.fill_diagonal(A, A.diagonal() - Ce2/Tt*r/fs)
    
    # Right-hand-side
    Ce1 = 1.4*(1 + 0.045*cp.sqrt(k/v2))
    b = -1/Tt*Ce1*Pk
    # pdb.set_trace()
    # Wall boundary conditions
    e[0 ] = mu[0 ]*k[1 ]/r[0 ]/cp.power(mesh.z[1 ]-mesh.z[0 ], 2)
    # e[-1] = mu[-1]*k[-2]/r[-1]/cp.power(mesh.z[-1]-mesh.z[-2], 2)
    # Solve
    e = Solver.solveEqn2(e*fs, A, b[1:p.nz-1], underrelaxE,p.igmres,mesh)/fs
    # e = Solver.solveEqn(e*fs, A, b[1:p.nz-1], underrelaxE,p.igmres)/fs
    e[1:p.nz-1] = cp.maximum(e[1:p.nz-1], 1.e-12)

    
    # ---------------------------------------------------------------------
    # k-equation
    
    # effective viscosity

    mueff = (mu + mut/sigk)/cp.sqrt(r)
    fs    = r
    fd    = 1/cp.sqrt(r)

    
    # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
    A = cp.einsum('i,ij->ij', mueff*fd, mesh.d2dz2.toarray()) \
      + cp.einsum('i,ij->ij', (mesh.ddz@mueff)*fd, mesh.ddz.toarray())
    
    # implicitly treated source term
    cp.fill_diagonal(A, A.diagonal() - r*e/k/fs)

    # Right-hand-side
    b = -Pk
    
    
    # Wall boundary conditions
    k[0] = k[-1] =  small
    # pdb.set_trace()
    # Solve
    # k = Solver.solveEqn2(k*fs, A, b[1:-1], underrelaxK,p.igmres,mesh)/fs
    k = Solver.solveEqn(k*fs, A, b[1:-1], underrelaxK,p.igmres)/fs
    k[1:-1] = cp.maximum(k[1:-1], 1.e-12)
    # pdb.set_trace()
    return mut,k,e,v2

