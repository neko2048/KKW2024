program test
use mpi
use pnetcdf
use petsc_vvm_solver
use petscsys
use netcdf, only: nf90_strerror
implicit none

integer :: i, j, k, sigmaIdx
integer :: mpierr, nproc
!REAL :: tarray(2), time, dtime
INTEGER :: initTimeIdx, endTimeIdx, idxInterval, timeIdx
REAL :: startCPUTime, endCPUTime
REAL :: dum1, dum2
character(256) :: outputPath, inputPath, fortPath, fname
integer :: ncid1, varid1, dimid1, dimid2, dimid3, dimid4
INTEGER(KIND = mpi_offset_kind),  DIMENSION(4) :: arraystart, readlen
REAL(kind = dbl_kind), dimension(:, :, :), allocatable :: f, vrtAcc3D
REAL(kind = dbl_kind), dimension(:, :, :, :), allocatable :: vrtAcc4D
integer(MPI_offset_kind) :: temp
INTEGER, DIMENSION(1) :: timeIdxArray  = (/ 400 /) ! ###
CHARACTER(256), DIMENSION(2) :: sigmaArray = (/ '1.5', '4.0'/) ! ###


CALL PetscInitialize(PETSC_NULL_CHARACTER, mpierr)
call mpi_comm_size(mpi_comm_world, nproc, mpierr)
call mpi_comm_rank(mpi_comm_world, my_task, mpierr)

100 FORMAT(A,I6.6,A)
200 FORMAT(2A)
300 FORMAT(2A,I6.6,A)
400 FORMAT(A, F7.1, A)
500 FORMAT(3A)

nsbdm_x = int(sqrt(dble(nproc)))
nsbdm_y = int(sqrt(dble(nproc)))

mi1 = mi_glob/nsbdm_x
mj1 = mj_glob/nsbdm_y

ni_sbdm = mod(my_task, nsbdm_x)
nj_sbdm = my_task/nsbdm_x

dx = 100. ! ### 
dynew = 100. ! ###
dz = 150. ! ###

if (my_task == 0) write(*, *) "Start Read fort.98"
fortPath = "/data/atmenu10246/convBuoy/solverOfAcce/dat/jet/mjo_std_mg/" ! ### and read 
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
  IF (sigmaArray(sigmaIdx) == '0.0') THEN
    WRITE(inputPath, 500) "/data/atmenu10246/convBuoy/solverOfAcce/dat/jet/", & ! ###
                          "laplacianBuoyancy/", "uniform-1/"
    WRITE(outputPath, 500) "/data/atmenu10246/convBuoy/solverOfAcce/dat/jet/", &
                           "vertAcce/", "uniform-1/"
  ELSE
    WRITE(inputPath, 500) "/data/atmenu10246/convBuoy/solverOfAcce/dat/jet/laplacianBuoyancy/gaussian-", trim(sigmaArray(sigmaIdx)), "/"
    WRITE(outputPath, 500) "/data/atmenu10246/convBuoy/solverOfAcce/dat/jet/vertAcce/gaussian-", trim(sigmaArray(sigmaIdx)), "/"
  ENDIF
  DO i = 1, size(timeIdxArray)
    timeIdx = timeIdxArray(i)
    allocate(f(1:mi1, 1:mj1, 1:nk2), vrtAcc3D(0:mi1+1, 0:mj1+1, 1:nk2), vrtAcc4D(0:mi1+1, 0:mj1+1, 1:nk2, 1))
    if (my_task == 0) WRITE(*, 100) "-------------------- ", timeIdx, " --------------------"
    WRITE(fname, 300) TRIM(inputPath),"buoyancy-",timeIdx,".nc"
    if (my_task == 0) WRITE(*, 200) "INPUT: ", fname
  
    
    mpierr=nfmpi_open(mpi_comm_world, trim(fname), nf_nowrite, mpi_info_null, ncid1)
    if (mpierr/=nf_noerr) write(*,*) "open fail", nf90_strerror(mpierr)
  
    arraystart = (/ NI_SBDM*MI1+1, NJ_SBDM*MJ1+1, 1, 1 /)
    readlen    = (/ MI1, MJ1, NK2, 1 /)
  
    mpierr = nfmpi_inq_varid(ncid1, 'buoyancy', varid1)
    IF (mpierr /= nf_noerr) stop "inq buoyancy"
    mpierr = nfmpi_get_vara_double_all(ncid1, varid1, arraystart, readlen, f(1:MI1, 1:MJ1, 1:nk2))
    IF (mpierr /= nf_noerr) stop "read buoyancy"
    mpierr = nfmpi_close(ncid1)
    if (mpierr /= nf_noerr) stop "close fail"
    !if (my_task == 0) write(*, *) "Buoy Read"
  
  
    call cpu_time(startCPUTime)
    if (my_task == 0) write(*, *) "Start to solve"
  
    do k=1, nk2
      f(:, :, k)=f(:, :, k)*rhou(k)
    enddo
    call petsc_solve_3d(f(1:mi1, 1:mj1, 2:nk2), vrtAcc3D(0:mi1+1, 0:mj1+1, 1:nk2))
    ! ### topo
    vrtAcc3D(:, :, 1)=0.
    
    call cpu_time(endCPUTime)
    if (my_task == 0) write(*, 400) "End solving | Cost Time: ", endCPUTime - startCPUTime, " sec"
    
    WRITE(fname, 300) TRIM(outputPath),"a-",timeIdx,".nc"
    mpierr = nfmpi_create(mpi_comm_world, trim(fname), or(nf_clobber, nf_64bit_offset), mpi_info_null, ncid1)
    if (mpierr /= nf_noerr) write(*, *) "create fail"
  
    mpierr = nfmpi_def_dim(ncid1, 'xc', mi_glob, dimid1)
    mpierr = nfmpi_def_dim(ncid1, 'yc', mj_glob, dimid2)
    mpierr = nfmpi_def_dim(ncid1, 'zc', nk2, dimid3)
    mpierr = nfmpi_def_dim(ncid1, 'time', 1, dimid4)
    mpierr = nfmpi_def_var(ncid1, 'a', nf_double, 4, (/dimid1, dimid2, dimid3, dimid4/), varid1)
    mpierr = nfmpi_enddef(ncid1)
  
    arraystart = (/ NI_SBDM*MI1+1, NJ_SBDM*MJ1+1, 1, 1 /)
    readlen    = (/ MI1, MJ1, NK2, 1 /)
  
    vrtAcc4D(:, :, :, 1) = vrtAcc3D
    mpierr = nfmpi_put_vara_double_all(ncid1, varid1, arraystart, readlen, vrtAcc4D(1:mi1, 1:mj1, 1:nk2, :))
    if (mpierr /= nf_noerr) write(*, *) nfmpi_strerror(mpierr)
    IF (mpierr /= nf_noerr) stop "nfmpi_strerror"
  
    mpierr = nfmpi_close(ncid1)
    if (mpierr /= nf_noerr) stop "close fail"
    if (my_task == 0) WRITE(*, 200) "OUTPUT: ", fname
    deallocate(f, vrtAcc3D, vrtAcc4D)
  ENDDO
ENDDO
CALL PETSCFINALIZE(mpierr)

end program test
