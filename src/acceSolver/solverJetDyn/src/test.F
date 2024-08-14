program test
use mpi
use pnetcdf
use petsc_vvm_solver
use petscsys
use netcdf, only: nf90_strerror
implicit none

integer :: i, j, k, sigmaIdx, varIdx
integer :: mpierr, nproc
!REAL :: tarray(2), time, dtime
INTEGER :: initTimeIdx, endTimeIdx, idxInterval, timeIdx
REAL :: startCPUTime, endCPUTime
REAL :: dum1, dum2
character(256) :: outputPath, inputPath, fortPath, fname, varName
integer :: ncid1, varid1, dimid1, dimid2, dimid3, dimid4
INTEGER(KIND = mpi_offset_kind),  DIMENSION(4) :: arraystart, readlen
REAL(kind = dbl_kind), dimension(:, :, :), allocatable :: f, vrtAcc3D
REAL(kind = dbl_kind), dimension(:, :, :, :, :), allocatable :: vrtAcc5D
integer(MPI_offset_kind) :: temp
INTEGER, DIMENSION(1) :: timeIdxArray  = (/ 460 /)
CHARACTER(256), DIMENSION(1) :: sigmaArray = (/ '0.0'/)
!CHARACTER(256), DIMENSION(13) :: sigmaArray = (/ '0.5' , '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '5.0', '7.5', '10.0', '15.0', '20.0'/)
CHARACTER(256), DIMENSION(5) :: varArray = (/ 'dm02', 'dm04', 'dm05', 'dm06', 'dm07' /)


CALL PetscInitialize(PETSC_NULL_CHARACTER, mpierr)
call mpi_comm_size(mpi_comm_world, nproc, mpierr)
call mpi_comm_rank(mpi_comm_world, my_task, mpierr)

100 FORMAT(A,I6.6,A)
200 FORMAT(2A)
300 FORMAT(2A,I6.6,A)
400 FORMAT(A, F7.1, A)
500 FORMAT(4A)

nsbdm_x = int(sqrt(dble(nproc)))
nsbdm_y = int(sqrt(dble(nproc)))

mi1 = mi_glob/nsbdm_x
mj1 = mj_glob/nsbdm_y

ni_sbdm = mod(my_task, nsbdm_x)
nj_sbdm = my_task/nsbdm_x

dx = 100.
dynew = 100.
dz = 150.

if (my_task == 0) write(*, *) "Start Read fort.98"
fortPath = "/data/atmenu10246/convBuoy/solverOfAcce/dat/jet/mjo_std_mg/"
fname=trim(fortPath)//"fort.98"
open(10, file=trim(fname))
DO i=1, 188
  READ(10, *)
ENDDO
DO i=1, nk2+1
  READ(10, *) j,  dum1, dum2, fnz(i), fnu(i)
ENDDO
DO i=1, 3
  READ(10, *)
ENDDO
DO i=1, nk2+1
  READ(10, *) j,  rhou(i)
ENDDO
DO i=1, 3
  READ(10, *)
ENDDO
DO i=1, nk2+1
  READ(10, *) j
ENDDO
DO i=1, 3
  READ(10, *)
ENDDO
DO i=1, nk2
  READ(10, *) j,  rhoz(i)
ENDDO
if (my_task == 0) write(*, *) "fort.98 Read Successfully"
call petsc_solver_initialize(0)

DO sigmaIdx = 1, size(sigmaArray)
  
  WRITE(inputPath, 500) "/data/atmenu10246/convBuoy/solverOfAcce/dat/yhkTaper/", & 
                        "dynTerms/gaussian-", trim(sigmaArray(sigmaIdx)), "/"
  WRITE(outputPath, 500) "/data/atmenu10246/convBuoy/solverOfAcce/dat/yhkTaper/", &
                         "dynTermsAcce/gaussian-", trim(sigmaArray(sigmaIdx)), "/"

  
  DO i = 1, size(timeIdxArray)
    timeIdx = timeIdxArray(i)
    allocate(f(1:mi1, 1:mj1, 1:nk2), vrtAcc3D(0:mi1+1, 0:mj1+1, 1:nk2), vrtAcc5D(0:mi1+1, 0:mj1+1, 1:nk2, 1, 5))
    if (my_task == 0) WRITE(*, 100) "-------------------- ", timeIdx, " --------------------"
    WRITE(fname, 300) TRIM(inputPath),"diag-",timeIdx,".nc"
    if (my_task == 0) WRITE(*, 200) "INPUT: ", fname
  
    mpierr=nfmpi_open(mpi_comm_world, trim(fname), nf_nowrite, mpi_info_null, ncid1)
    if (mpierr/=nf_noerr) write(*,*) "open fail", nf90_strerror(mpierr)
  
    arraystart = (/ NI_SBDM*MI1+1, NJ_SBDM*MJ1+1, 1, 1 /)
    readlen    = (/ MI1, MJ1, NK2, 1 /)
    DO varIdx = 1, size(varArray)
      varName = varArray(varIdx)
      mpierr = nfmpi_inq_varid(ncid1, varName, varid1)
      IF (mpierr /= nf_noerr) stop "inq fail"
      mpierr = nfmpi_get_vara_double_all(ncid1, varid1, arraystart, readlen, f(1:MI1, 1:MJ1, 1:nk2))
      IF (mpierr /= nf_noerr) stop "read fail"
  
      call cpu_time(startCPUTime)
      if (my_task == 0) write(*, *) "Start to solve: ", trim(varName)
  
      do k=1, nk2
        f(:, :, k)=f(:, :, k)*rhou(k)
      enddo
      call petsc_solve_3d(f(1:mi1, 1:mj1, 2:nk2), vrtAcc3D(0:mi1+1, 0:mj1+1, 1:nk2))
      vrtAcc3D(:, :, 1)=0.
      call cpu_time(endCPUTime)
      if (my_task == 0) write(*, 400) "End solving | Cost Time: ", endCPUTime - startCPUTime, " sec"
  
      vrtAcc5D(:, :, :, 1, varIdx) = vrtAcc3D
      vrtAcc3D = 0.
      f = 0.
    ENDDO
    !===== close input ======!
    mpierr = nfmpi_close(ncid1)
    if (mpierr /= nf_noerr) stop "close fail"
  
    WRITE(fname, 300) TRIM(outputPath),"a-",timeIdx,".nc"
    mpierr = nfmpi_create(mpi_comm_world, trim(fname), or(nf_clobber, nf_64bit_offset), mpi_info_null, ncid1)
    if (mpierr /= nf_noerr) write(*, *) "create fail"
  
    mpierr = nfmpi_def_dim(ncid1, 'xc', mi_glob, dimid1)
    mpierr = nfmpi_def_dim(ncid1, 'yc', mj_glob, dimid2)
    mpierr = nfmpi_def_dim(ncid1, 'zc', nk2, dimid3)
    mpierr = nfmpi_def_dim(ncid1, 'time', 1, dimid4)
    arraystart = (/ NI_SBDM*MI1+1, NJ_SBDM*MJ1+1, 1, 1 /)
    readlen    = (/ MI1, MJ1, NK2, 1 /)
      
      mpierr = nfmpi_enddef(ncid1)
    DO varIdx = 1, size(varArray)
      varName=varArray(varIdx)
      if (my_task == 0) write(*, *) "save", trim(varName)
      mpierr = nfmpi_redef(ncid1)
      mpierr = nfmpi_def_var(ncid1, trim(varName), nf_double, 4, (/dimid1, dimid2, dimid3, dimid4/), varid1)
      mpierr = nfmpi_enddef(ncid1)
      mpierr = nfmpi_put_vara_double_all(ncid1, varid1, arraystart, readlen, vrtAcc5D(1:mi1, 1:mj1, 1:nk2, :, varIdx))
      if (mpierr /= nf_noerr) then
        write(*, *) nfmpi_strerror(mpierr)
        stop "nfmpi_strerror"
      endif
    ENDDO
    !mpierr = nfmpi_enddef(ncid1)
  
    mpierr = nfmpi_close(ncid1)
    if (mpierr /= nf_noerr) stop "close fail"
    if (my_task == 0) WRITE(*, 200) "OUTPUT: ", fname
    deallocate(f, vrtAcc3D, vrtAcc5D)
  ENDDO
ENDDO
CALL PETSCFINALIZE(mpierr)

end program test
