def solveEqn(x,A,b,omega):
    import cupy as cp
    n = cp.size(x)
    x_new = x.copy()
    # add boundary conditions
    b = b - x[0]*A[1:n-1,0] - x[n-1]*A[1:n-1,n-1]
    
    # perform under-relaxation
    b[:] = b[:] + (1-omega)/omega * A.diagonal()[1:-1]*x[1:-1]
    cp.fill_diagonal(A, A.diagonal()/omega)
    
    # solve linear system
    x_new[1:-1] = cp.linalg.solve(A[1:-1, 1:-1], b)
    return x_new