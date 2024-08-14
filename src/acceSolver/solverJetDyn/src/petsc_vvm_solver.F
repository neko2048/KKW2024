#include "petsc/finclude/petscmat.h"
#include "petsc/finclude/petscksp.h"

MODULE petsc_vvm_solver
! Apply PETsc to solve 3D and 2D Poisson equation in vvm.
! 
! The 3D equation is W-equation or Continuity eqaution for vertical velocity(2.19).
! The 2D equation are the top boundary condtion for psi(2.23) and chi(2.24). 
! 
! In this subroutine, we name the variables relating to solver setting-up based on the equations.
!
! For example, ksp_Conti is the Krylov methods (Martix solver) for Continuity equation.
!
! W equation   (2.19): Conti
! Psi equation (2.23): TopPsi
! Chi equation (2.24): TopChi
!
! Basiclly, we have three parameter can choose for each solver.
!
! Method          : ksp_type_Conti, ksp_type_TopPsi, ksp_type_TopChi
! Preconditioning : PC_type_Conti, PC_type_TopPsi, PC_type_TopChi
!
! Variable for petsc_vvm_solver to choose ksp solver and preconditioning for equations.
! Possible method can be choosed from
! http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPType.html#KSPType
! 
! The recommended method is KSPCG for SPD (symmetric positive definite) systems.
! (System in Jung 2007 is SPD)
!
! Other methods such like KSPBCGS and KSPGMRES can be applied for present system.
!
! If you change the system in "ComputeConti", you should check whether your system keep the 
! SPD property. If not, please use KSPBCGS or GMRES instead. 
!
! The possible preconditioning can be choose from following website.
! http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/PC/PCType.html#PCType
! 
! Precondtioning can improve the efficiency for KSP system.
!
! If you are not familiar with Preconditioning, you can choose PCNONE for preconditioning-free.
!
! Our recommend method are PCBJACOBI and PCEISENSTAT.
!
! For the PCBJACOBI, it can reduce number of iteration as well as keep consistant on parallization 
! boundary.
!
! For the PCEISENSTAT, you must set the Omega (similar in SSOR), the efficiency is better then
! PCBJACOBI, but there exsist flaw around the parallelization boundary with respect to your 
! tolerance. (Omega default 1.65d0)
!
!-------------------------------------------------------------------------------------------
! Tolerances 	  : Tol_Conti, Tol_TopPsi, Tol_TopChi
!                  aTol_Conti, aTol_TopPsi, aTol_TopChi  
!                  dTol_Conti, dTol_TopPsi, dTol_TopChi
!
!  Tol = the relative convergence tolerance (relative decrease in the residual norm)
! aTol = the absolute convergence tolerance (absolute size of the residual norm)
! dTol = the divergence tolerance concludes that the method is diverging)
!
! The tolerance setting is quite difference for Conti, TopChi and TopPsi.
! For Conti and TopChi, model satbility is determined by aTol(absolute convergence tolerance).
! For TopPsi, it is determined by Tol(relative convergence tolerance), since it is relatively 
! small when simulation begining.
!
! Recommendied tolerance for test or experiment 
!           Tol/atol/dtol for Conti are 1.d-15/1.d-8/1.d2
!                         for TopPsi are 1.d-4/1.d-30/1.d2 
!                         for TopChi are 1.d-15/1.d-4/1.d2
! For the efficiency, the optimized iteration is about 10 to 30, so you can adjust the tolerance
! with respect to your experiments.
!
! Recommendied tolerane for production
!           Tol/atol/dtol for Conti are 1.d-30/1.d-12/1.d2
!                         for TopPsi are 1.d-9/1.d-30/1.d2
!                         for TopChi are 1.d-15/1.d-9/1.d2
!
! MXIT is maximum iteration for PETsc ksp solver. The recommend value is 1000.
!
!---------------------------------------------------------------------------
!
! The differential system is inport to petsc_vvm_solver instead of direct_3d and direct_xy.
!
! If you wish to modify the differential coefficient and system, you can modify the subroutine
!
! ComputeConti(ksp,JJ,jac,str,ctx,ierr) and ComputeTopbc(ksp,JJ,jac,str,ctx,ierr).
!
! The instruction can be refer on the website of PETsc. http://www.mcs.anl.gov/petsc/
!
! Or just change the coiefficient of an,bn,cn, dx and dynew with respect to their location.
!
!---------------------------------------------------------------------------
!
! Present computational grid in based on PETsc vec system, vector(w, psi and chi) couple with
! PETsc variable system must use subroutine VecGetArray and VecRestoreArray. It can't be used 
! directly by fortran matrix.
!
! The distributed data for parallel computing is allocated by dmda system, our input is global
! dimension of computational grid and cores. You can modify the global dimension or local dimension
! by change the input of DMDACreate3d or DMDACreate2d.

USE petscsys
USE petscvec
USE petscmat
USE petscksp
USE petscpc
USE petscdm
USE petscdmda

IMPLICIT NONE
PRIVATE
integer,parameter,public :: mi_glob=1024, mj_glob=1024, nk2=120, nhalo=1
INTEGER, PARAMETER,public :: &
    char_len  = 80, &
    log_kind  = KIND(.TRUE.), &
    int_kind  = SELECTED_INT_KIND  (09), &
    real_kind = SELECTED_REAL_KIND (06), &
    dbl_kind  = SELECTED_REAL_KIND (13)
real(kind=dbl_kind), public :: dx, dynew, dz

integer,public :: mi1,mj1,nsbdm_x,nsbdm_y,ni_sbdm,nj_sbdm
integer,public :: my_task
real(kind=dbl_kind),dimension(nk2+1),public :: rhoz,rhou,fnu,fnz

PetscInt, PRIVATE:: i,j,k,kt,npt,np
PetscInt, PRIVATE:: ixx,iy,iz,mx,my,mz,ixg,iyg,izg,mxg,myg,mzg
PetscInt, PRIVATE:: ctx3,ctx2
PetscScalar, PRIVATE:: tmp(1)
PetscScalar, PRIVATE:: vsum,tpts = -1.d0/dble(mi_glob*mj_glob)
PetscOffset, PRIVATE:: ip
PetscErrorCode, PRIVATE:: ierr
Vec, PRIVATE::v,vl,rhs,v1,v2
PC, PRIVATE::PC_Conti,PC_Toppsi,PC_Topchi
KSP Conti,Toppsi,Topchi
PetscInt, PARAMETER, PRIVATE:: nlevel = 1, nlv = 0
Character(len=*), PARAMETER, PRIVATE :: ksp_type_conti = KSPCG, pc_type_Conti = PCEISENSTAT
!Character(len=*), PARAMETER, PRIVATE :: ksp_type_toppsi = KSPCG, pc_type_toppsi = PCEISENSTAT
!Character(len=*), PARAMETER, PRIVATE :: ksp_type_topchi = KSPCG, pc_type_topchi = PCEISENSTAT
!sub-ksp and respect pc for PCMG setting

DM, PRIVATE::da3(nlevel),da2(nlevel)
Mat, PRIVATE:: R, Amat,Cmat
PetscReal, PARAMETER, PRIVATE:: Omega =1.65d0
! Set tolerance  
PetscInt, PRIVATE:: MXIT = 2000
PetscReal, PRIVATE:: tol_Conti = 1.d-15,atol_Conti=1.d-7,dtol_Conti = 1.d2
! atol_Conti=1.d-6 / 1.d-7?
!PetscViewer viewer

PUBLIC :: petsc_solver_initialize,petsc_solver_finalize,petsc_solve_3d
CONTAINS
SUBROUTINE petsc_solver_initialize(petsc_cr)

!-----------------------------------------------------------------------

! Create 3d grid by DMDACreate3d 
! http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DM/DMDACreate3d.html
! Get the grid information about grid size @ global by
! DMDAGetCorners and DMDAGetGhostCorners
PetscInt, intent(in) :: petsc_cr

! set additional criterion if petsc_cr == 1 
!if (petsc_cr .eq. 1) then
!MXIT = 2000
!tol_Conti = 1.d-15
!atol_Conti=1.d-10
!dtol_Conti = 1.d2
!endif

    call DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,                       &
      DM_BOUNDARY_PERIODIC,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,MI_glob/(2**nlv), &
      MJ_glob/(2**nlv),NK2-1,nsbdm_x,nsbdm_y,1,1,nhalo,PETSC_NULL_INTEGER,      &
      PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,da3(nlevel),ierr)
    call DMSetUp(da3(nlevel),ierr) ! must be call after PETSc 3.8.0

    call DMDASetUniformCoordinates(da3(nlevel),0.d0,1.d0,0.d0,1.d0,0.d0,1.d0,ierr)

    DO I = nlevel,2,-1
    call DMDAGetCorners(da3(I),ixx,iy,iz,mx,my,mz,ierr)
    call DMDASetRefinementFactor(da3(I),2,2,1,ierr)
    call DMRefine(da3(I), PETSC_COMM_WORLD, da3(I-1),ierr)
    call DMDASetUniformCoordinates(da3(I-1),0.d0,1.d0,0.d0,1.d0,0.d0,1.d0,ierr)
    ENDDO

    call DMDAGetCorners(da3(1),ixx,iy,iz,mx,my,mz,ierr)
    call DMDAGetGhostCorners(da3(1),ixg,iyg,izg,mxg,myg,mzg,ierr)
    !if (my_task==0) write(*,*) ixx,iy,iz,mx,my,mz

    npt=mi_glob*mj_glob*(NK2-1)
    np=nsbdm_x*nsbdm_y

    call MatCreateAIJ(MPI_COMM_WORLD,npt/np,npt/np,npt,npt,7, &
         PETSC_NULL_INTEGER,5,PETSC_NULL_INTEGER,Cmat,ierr)
    call OperatorConti(Cmat)

! http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSP.html#KSP
! Connect KSP with DM grid -- KSPSetDM
! Set KSP method type 
! http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPType.html#KSPType
!
! Get KSP preconditioning and Set KSP preconditioning
! http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/PC/PCType.html#PCType
! Present PC is SOR (Successive over-relaxation) with Omega = 1.65
!
! Tolorence set with 3 type. KSPSetTolerances
! http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPSetTolerances.html 
!

    call KSPCreate(MPI_COMM_WORLD,Conti,ierr)
!    call KSPSetDM(Conti,da3,ierr)
!    call KSPSetComputeOperators(Conti,ComputeConti,ctx3,ierr)    
    call KSPSetOperators(Conti,Cmat,Cmat,ierr)
    call KSPSetType(Conti,ksp_type_conti,ierr)
    call KSPGetPC(Conti,PC_Conti,ierr)
    call PCSetType(PC_Conti,pc_type_conti,ierr)
    IF (pc_type_conti .eq. PCEISENSTAT) call PCEisenstatSetOmega(PC_Conti,Omega,ierr)
    call KSPSetFromOptions(Conti,ierr)
    call KSPSetTolerances(Conti,tol_Conti,atol_Conti,dtol_Conti,mxit,ierr)
! Create a global vector and set initial guess as following
! KSPSetInitialGuessNonzero(Conti,PETSC_TRUE,ierr)
 
    call DMCreateGlobalVector(da3(1),v,ierr)
    call DMCreateGlobalVector(da3(1),rhs,ierr)
    call VecSet(rhs,0.d0,ierr)

    call KSPSolve(Conti,rhs,v,ierr)
    call KSPSetInitialGuessNonzero(Conti,PETSC_TRUE,ierr)
    call VecDestroy(rhs,ierr)


END SUBROUTINE petsc_solver_initialize





SUBROUTINE petsc_solver_finalize

!    DO I=1,2*nlevel-1
!    call KSPDestroy(sksp_Conti(I),ierr)
!    call KSPDestroy(sksp_toppsi(I),ierr)
!    call KSPDestroy(sksp_topchi(I),ierr)
!    enddo
    call KSPDestroy(Conti,ierr)
    call MatDestroy(Cmat,ierr)
    !call KSPDestroy(Toppsi,ierr)
    !call KSPDestroy(Topchi,ierr)
    DO I=1,nlevel
    call DMDestroy(da3(i),ierr)
    call DMDestroy(da2(i),ierr)
    enddo
END SUBROUTINE petsc_solver_finalize





SUBROUTINE petsc_solve_3d(F,X)
REAL (KIND=dbl_kind), INTENT(INOUT)::F(mi1,mj1,nk2-1)
REAL (KIND=dbl_kind), INTENT(INOUT)::X(mi1+2,mj1+2,nk2)
PetscInt its
PetscReal norm
    call DMCreateGlobalVector(da3(1),rhs,ierr)
    call DMCreateLocalVector(da3(1),vl,ierr)

      call VecGetArray(vl,tmp,ip,ierr)
      do k=iz,iz+mz-1
      do j=iy,iy+my-1
      do i=ixx,ixx+mx-1
      kt = i - ixg + (j - iyg)*mxg + (k - izg)*mxg*myg + 1
      tmp(ip+kt)=-F(I-ixx+1,J-iy+1,K+1)
      !if (my_task==0) write(*,*) -F(I-ixx+1,J-iy+1,K+1)
      enddo
      enddo
      enddo
      call VecRestoreArray(vl,tmp,ip,ierr)

      call DMLocalToGlobalBegin(da3(1),vl,INSERT_VALUES,rhs,ierr)
      call DMLocalToGlobalEnd(da3(1),vl,INSERT_VALUES,rhs,ierr)

      call KSPSetFromOptions(Conti,ierr)
      call KSPSolve(Conti,rhs,v,ierr)

      IF ( MY_TASK .EQ. 0) THEN
      call KSPGetIterationNumber(Conti,its,ierr)
      call KSPGetResidualNorm(Conti,norm,ierr)
      write(6,100) norm,its
      ENDIF
 100  format('Norm of error = ',e11.4,',  Iterations = ',i5)

      call DMGlobalToLocalBegin(da3(1),v,INSERT_VALUES,vl,ierr)
      call DMGlobalToLocalEnd(da3(1),v,INSERT_VALUES,vl,ierr)

      call VecGetArray(vl,tmp,ip,ierr)
      do k=iz,iz+mz-1
      do j=iy,iy+my-1
      do i=ixx,ixx+mx-1
      kt = i - ixg + (j - iyg)*mxg + (k - izg)*mxg*myg + 1
      X(I-ixx+2,J-iy+2,K+1) = tmp(ip+kt) / RHOU(K+1)
      !if (my_task==0) write(*,*) X(I-ixx+2,J-iy+2,K+2) 
      enddo
      enddo
      enddo
      call VecRestoreArray(vl,tmp,ip,ierr)

    call VecDestroy(rhs,ierr)
    call VecDestroy(vl,ierr)

END SUBROUTINE petsc_solve_3d





SUBROUTINE OperatorConti(jac)

      Mat         jac
      PetscInt     ltog(1)
      PetscOffset idltog,idx
      PetscInt grow(1)
      PetscInt row,i1,i7,i6,irow,jrow,krow
      PetscInt col(7),iseven
      PetscScalar two,one,lambda,c1,c2,ctmp,tem,dzsq
      PetscScalar vv(7),AN(NK2-1),BN(NK2-1),CN(NK2-1),B_loc(NK2),BU(NK2)
      ISLocalToGlobalMapping ltogm

      one    = 1.
      two    = 2.

      i1 = 1
      i7 = 7
      i6 = 6
      c1 = -1.d0/(DX*DX)
      c2 = -1.d0/(DYNEW*DYNEW)
      ctmp = -2.d0*(c1+c2)

      !do K = 1, Nk2-2
      !B_loc(K) = 1. / ( RHOZ(K+1) * FNZ(K+1) )
      !BU(K) = FNU(K+1) / RHOU(K+1)
      !enddo
      !BU(Nk2-1) = FNU(Nk2) / RHOU(Nk2)

      !DZSQ = DZ*DZ
      !do K = 1, Nk2-2
      !TEM = 1. / ( B_loc(K) * DZSQ )
      !AN(K) = - BU(K) * TEM
      !CN(K) = - BU(K+1) * TEM
      !BN(K) = -( AN(K) + CN(K) )
      !if (my_task==0) write(*,'(A,I4,3F15.10)') "old,",k,an(k),bn(k),cn(k)
      !enddo
      !AN(1) = 0.
      !CN(Nk2-2) = 0.

      do K = 1, Nk2-1
      B_loc(K) = 1. / ( RHOU(K+1) * FNU(K+1) )
      BU(K) = FNZ(K) / RHOZ(K)
      enddo
      BU(Nk2) = FNU(Nk2) / RHOU(Nk2)

      DZSQ = DZ*DZ
      do K = 1, Nk2-1
      TEM = 1. / ( B_loc(K) * DZSQ )
      AN(K) = - BU(K) * TEM
      CN(K) = - BU(K+1) * TEM
      BN(K) = -( AN(K) + CN(K) )
      enddo
      AN(1) = 0.
      CN(Nk2-1) = 0.


      call DMGetLocalToGlobalMapping(da3(1),ltogm,ierr)
      call ISLocalToGlobalMappingGetIndices(ltogm,ltog,idltog,ierr)


      k=iz
        row = (k - izg)*mxg*myg !+ (ys-gys)*gxm !+  xs - gxs
        do  j=iy,iy+my-1
          irow = row + (j - iyg)*mxg + ixx - ixg
          do i=ixx,ixx+mx-1
            krow = irow + (i - ixg)
            grow(1) = ltog(idltog+krow)
!          print*,i,j,k,krow,ltog(idltog+krow)
            vv(1)   = c2
            col(1) = ltog(idltog+krow - mxg)
            vv(2)   = c1
            col(2) = ltog(idltog+krow - 1)
            vv(3)   = ctmp + bn(k+1)
            col(3) = grow(1)
            vv(4)   = c1
            col(4) = ltog(idltog+krow + 1)
            vv(5)   = c2
            col(5) = ltog(idltog+krow + mxg)
            vv(6)   = cn(k+1)
            col(6) = ltog(idltog+krow + mxg*myg)
      call MatSetValues(jac,i1,grow,i6,col,vv,INSERT_VALUES,ierr)
          enddo
        enddo

      do k=iz+1,iz+mz-2
        row = (k - izg)*mxg*myg !+ (ys-gys)*gxm !+  xs - gxs
        do  j=iy,iy+my-1
          irow = row + (j - iyg)*mxg + ixx - ixg
          do i=ixx,ixx+mx-1
            krow = irow + (i - ixg)
            grow(1) = ltog(idltog+krow)
!          print*,i,j,k,krow,ltog(idltog+krow)
            vv(1)   = an(k+1)
            col(1) = ltog(idltog+krow - mxg*myg)
            vv(2)   = c2
            col(2) = ltog(idltog+krow - mxg)
            vv(3)   = c1
            col(3) = ltog(idltog+krow - 1)
            vv(4)   = ctmp + bn(k+1)
            col(4) = grow(1)
            vv(5)   = c1
            col(5) = ltog(idltog+krow + 1)
            vv(6)   = c2
            col(6) = ltog(idltog+krow + mxg)
            vv(7)   = cn(k+1)
            col(7) = ltog(idltog+krow + mxg*myg)
      call MatSetValues(jac,i1,grow,i7,col,vv,INSERT_VALUES,ierr)
          enddo
        enddo
      enddo

      k=mz-1
        row = (k - izg)*mxg*myg !+ (ys-gys)*gxm !+  xs - gxs
        do  j=iy,iy+my-1
          irow = row + (j - iyg)*mxg + ixx - ixg
          do i=ixx,ixx+mx-1
            krow = irow + (i - ixg)
            grow(1) = ltog(idltog+krow)
!          print*,i,j,k,krow,ltog(idltog+krow)
            vv(1)   = an(k+1)
            col(1) = ltog(idltog+krow - mxg*myg)
            vv(2)   = c2
            col(2) = ltog(idltog+krow - mxg)
            vv(3)   = c1
            col(3) = ltog(idltog+krow - 1)
            vv(4)   = ctmp + bn(k+1)
            col(4) = grow(1)
            vv(5)   = c1
            col(5) = ltog(idltog+krow + 1)
            vv(6)   = c2
            col(6) = ltog(idltog+krow + mxg)
      call MatSetValues(jac,i1,grow,i6,col,vv,INSERT_VALUES,ierr)
          enddo
        enddo

      call ISLocalToGlobalMappingRestoreIndices(ltogm,ltog,idltog,ierr)

      call MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr)
      return


END SUBROUTINE OperatorConti

END MODULE petsc_vvm_solver
