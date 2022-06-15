program main
    use eqsolve
    use global
implicit none
integer :: it

call allocateArrays

call init_grid()
call init_var()
do it = 1, itend
    ! call eqsolve()
    ! call checkRes()
end do

!-----------------------------------------------!
!-----------------------------------------------!
!                    contains clause            !
!-----------------------------------------------!
contains

!-----------------------------------------------!
!            init_field                         !
!-----------------------------------------------!
subroutine init_grid()
use global
integer :: dumbct,datct,iend,icnt,m,kind
integer,parameter :: datlen = 310
double precision :: tmpdat(28)
double precision, allocatable, dimension(:) :: utmp,ztmp
character(len=80) :: flnm, datdir
character(len=200) :: fulldir

allocate(utmp(datlen),ztmp(datlen))


!%%%%%%%%%%%%%%%%%%%%%%%%%
! move dir address as an input. makeshift operation for now
!%%%%%%%%%%%%%%%%%%%%%%%%%%    
datdir = '/home/hanlee/RANS_Dev/DNS_Data/'
flnm   = 'M8Tw048_Stat.dat'
fulldir = trim(datdir)//trim(flnm)
open(10,file=fulldir,form = 'formatted',status = 'old')
rewind 10
do dumbct = 1,142
    read(10,*)
enddo
iend = 0
icnt = 1
do while (iend .eq. 0)
    read(10 , *, iostat = iend) (tmpdat(m),m=1,28)
    utmp(icnt) = tmpdat(5)*Uinf
    ztmp(icnt) = tmpdat(1)
    icnt = icnt + 1 
end do

! borrowed from ref 1 in week 1 report k-w SST RANS case
z(1) = 0
z(2) = zpls2*ztau

! print*, '    zpls            z/delta_Ref'
! print*, z(1)/ztau, z(1)/delta_Ref
! print*, z(2)/ztau, z(2)/delta_Ref
do kind = 3, nz
z(kind) = (z(kind-1)-z(kind-2))*chi+z(kind-1)
print*, z(kind)/ztau, z(kind)/delta_Ref
enddo
! ddy=d/dchi*dchi/dz 
! 4th order forward difference for dchi/dz at the first grid point
dchidz(1) = 1/(-25/12*z(1)+4*z(2)-3*z(3)+4/3*z(4)-0.25*z(5))
do kind = 2, nz-1
    dchidz(kind) = 1/(-0.5*z(kind-1)+0.5*z(kind+1))
end do
dchidz(nz) = 1/(-25/12*z(nz)+4*z(nz-1)-3*z(nz-2)+4/3*z(nz-3)-0.25*z(nz-4))

deallocate(utmp,ztmp)
end subroutine init_grid


!-----------------------------------------------!
!              init_var                         !
!-----------------------------------------------!
subroutine init_var
    implicit none
    integer :: m 
    call init_u()
    call init_T()
    do m = 1,nz
        k = 0.01
        om = 1
    enddo
        
end subroutine init_var

!-----------------------------------------------!
!                init_u                         !
!-----------------------------------------------!
subroutine init_u()
    ! Reference : Wilcox, D.C. (2006) Turbulence Modeling for CFD. 3rd Edition, DCW Industries, Canada, CA, USA.
double precision, allocatable, dimension(:) :: upls_vis, upls_log
integer :: m
allocate(upls_vis(nz),upls_log(nz))
upls_vis = z/ztau
upls_log = 1/0.418*log(z/ztau)+5.2

upls = upls_vis+(1+tanh(0.3*(z/ztau-7)))/2*(upls_log-upls_vis)
upls(1) = 0
u(1)    = 0
do m = 2, nz
    u(m) = upls(m)*utau
    if ( u(m) .gt. Uinf ) u(m) = Uinf 
    ! print*, z(m)/ztau , upls(m)  , u(m)  
end do
deallocate(upls_log,upls_vis)
end subroutine init_u

!-----------------------------------------------!
!                init_T                         !
!-----------------------------------------------!
subroutine init_T()
    ! Reference : Duan, L., & Martín, M. (2011). Direct numerical simulation of hypersonic turbulent boundary layers.
    ! Part 4. Effect of high enthalpy. Journal of Fluid Mechanics, 684, 25-59. doi:10.1017/jfm.2011.252
implicit none
double precision :: func_u,Tr,Trat
integer :: m
Tr = Tinf*(1+0.9*(gam-1)/2*Minf**2)
do m = 1,nz
func_u = 0.1741*(u(m)/Uinf)**2 + 0.8259*(u(m)/Uinf)
T(m)   =Tinf*(Tw/Tinf+(Tr-Tw)/Tinf*func_u+(Tinf-Tr)/Tinf*(u(m)/Uinf)**2) 
print*, u(m), T(m)
end do

end subroutine init_T

subroutine allocateArrays
allocate(z(nz),u(nz), rho(nz), t(nz), om(nz), k(nz),upls(nz),mu(nz),dchidz(nz))
end subroutine allocateArrays

subroutine Linear1Dinterp(x1,x2,y1,y2,x,yout)
    real, intent(in) :: x1, x2, x
    real, intent(in) :: y1, y2
    real, intent(out) :: yout
    yout = (y2-y1)/(x2-x1)*(x-x1)+y1
end subroutine Linear1Dinterp

end program main