import cupy as cp
class initfield:
    def __init__(self,nz,ztau,utau,Uinf,Tinf,gam,Minf,Tw,z):
        # ----------------------------------------------------
        # Velocity Initialization
        # ----------------------------------------------------
        # upls_vis = z/utau
        # upls_log = 1/0.418*cp.log(z/ztau)+5.2

        # upls = upls_vis+(1+cp.tanh(0.3*(z/ztau-7)))/2*(upls_log-upls_vis)
        # upls[0] = 0

        u=cp.zeros(nz)
        # for m in range(1,nz):
        #     u[m] = upls[m]*utau
        #     if ( u[m] >= Uinf ):
        u[nz-1] = Uinf 
        self.u = u
        # ----------------------------------------------------
        # Temperature Initialization
        # ----------------------------------------------------
        Tr = Tinf*(1+0.9*(gam-1)/2*Minf**2)
        T=cp.zeros(nz)
        T[0] = Tw
        for m in range(1,nz):
            func_u = 0.1741*(u[m]/Uinf)**2 + 0.8259*(u[m]/Uinf)
            T[m]   = Tinf*(Tw/Tinf+(Tr-Tw)/Tinf*func_u+(Tinf-Tr)/Tinf*(u[m] /Uinf)**2) 
        self.T = T 

        # -------------------------------------------------------
        # density Initialization
        # -------------------------------------------------------
        rho = cp.zeros(nz)
        rwall = 0.0041
        for m in range(0,nz):
            rho[m] = rwall*Tw/T[m]
        self.rho = rho

        # -------------------------------------------------------
        # dynamic viscosity Initialization
        # -------------------------------------------------------
        mu = cp.zeros(nz)
        for m in range(0,nz):
            mu[m] = 1.458E-6*cp.sqrt(Tinf**3)/(Tinf+110.3)*(T[m]/Tinf)**0.76
        self.mu = mu        

        # ------------------------------------------------------
        # k & omega& mut Initialization
        # ------------------------------------------------------
        k = cp.zeros(nz)
        om =cp.zeros(nz) 
        mut = cp.zeros(nz)
        for m in range(0,nz):
            k[m] = 0.01
            om[m] = 1 
            mut[m] = 0 
        self.k = k
        self.om = om
        self.mut = mut

             