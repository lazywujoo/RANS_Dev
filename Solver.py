def solveEqn(x,A,b,omega,igmres):
    # import pdb

    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix
    import cupyx.scipy.sparse.linalg as cpx

    n = cp.size(x)
    x_new = x.copy()
    # add boundary conditions
    b = b - x[0]*A[1:n-1,0] - x[n-1]*A[1:n-1,n-1]
    # tmp2 = A[1:n-1,0]*x[0]
    # b[0] = b[0] - tmp2[0]
    # tmp2 = A[1:n-1,n-1]*x[n-1]
    # b[n-3] = b[n-3]- tmp2[n-3]
    A = csr_matrix(A)
    # perform under-relaxation
    # for m in range (0,n-3):   
    #     b[m] = b[m] + (1-omega)/omega * A[1:n-1,m+1]*x[1:n-1]
    if igmres == 0 : 
        # solve linear system
        x_new[1:-1] = cpx.spsolve(A[1:-1, 1:-1], b)
    elif igmres == 1:
        sol = cpx.gmres(A[1:n-1, 1:n-1],b,x[1:n-1], tol=1e-6, maxiter=1 )
        x_new[1:-1] = sol[0]
    return x_new

def solveEqn2(x,A,b,omega,igmres,mesh):
    # import pdb

    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix
    import cupyx.scipy.sparse.linalg as cpx
    import pdb

    n = cp.size(x)
    x_new = x.copy()
    # add boundary conditions
    b = b - x[0]*A[1:n-1,0] 
    A[n-2,:] = A[n-2,:] - mesh.ddz[n-2,:]*A[n-2,n-1]/mesh.ddz[n-2,n-1]
    A = csr_matrix(A)
    if igmres == 0 : 
        # solve linear system
        x_new[1:n-1] = cpx.spsolve(A[1:n-1, 1:n-1], b)
    elif igmres == 1:
        sol = cpx.gmres(A[1:n-1, 1:n-1],b,x[1:n-1], tol=1e-6, maxiter=3 )
        x_new[1:n-1] = sol[0]
    return x_new    