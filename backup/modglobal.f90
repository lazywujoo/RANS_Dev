module global
implicit none
double precision, parameter :: Uinf = 1155.1, utau = 54.3, zpls2=0.2, ztau = 67.7e-6
double precision, parameter :: chi = 1.025, delta_Ref = 0.0353,limres=1e-6,Tw = 298
double precision, parameter :: Tinf = 51.8, gam = 1.4 , Minf = 7.87, rinf = 0.026
integer, parameter :: itend = 2000,  nz=293
double precision, allocatable, dimension(:) :: z, u, rho, t, om, k,upls,mu,dchidz


end module global