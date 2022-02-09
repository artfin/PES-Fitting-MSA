import os
import logging
import shlex
from subprocess import Popen, PIPE, STDOUT

from dataset import PolyDataset

from collections import namedtuple
cmdstat = namedtuple('cmdstat', ['stdout', 'stderr', 'returncode'])

def cl(command):
    arg = shlex.split(command)

    p = Popen(command, stdout=PIPE, shell=True)
    stdout, stderr = p.communicate()

    stdout     = stdout.decode("utf-8").strip()
    returncode = p.returncode

    try:
        stderr = stderr.decode("utf-8").strip()
    except:
        stderr = ""

    return cmdstat(stdout, stderr, returncode)

def generate_fit_code(config_fname, NATOMS, POLY, NCONFIGS, a0=2.0, wt=1.0):
    return """
  program fit
  use basis
  implicit none

  external dgelss
  real :: HTOCM
  real :: rmse, wrmse, a0, dwt, vmin
  integer :: data_size
  real,allocatable::xyz(:,:,:),x(:)
  real,allocatable::v(:),b(:),p(:),wt(:)
  real,allocatable::yij(:,:), A(:,:)
  real,allocatable::coeff(:),v_out(:),s(:)
  real :: work(150000), dr(3)
  integer :: ncoeff, natm, ndis
  integer :: i, j, k, m, info, rank
  character(len=32) :: data_file
  character :: symb

  natm      = """ + str(NATOMS)   + """ ! change to the number of atoms
  ncoeff    = """ + str(POLY)     + """ ! change to the number of coeff. (size of c in bemsa.f90)
  data_size = """ + str(NCONFIGS) + """ ! change to the number of data points in pts.dat
  a0        = """ + str(a0)       + """
  dwt       = """ + str(wt)       + """

  ndis=natm*(natm-1)/2

  open(10,file="coeff.dat",status='unknown')
  open(11,FILE="points.eng",status='unknown')
  open(12,file=\"""" + config_fname + """\",status='old')

  allocate(x(ndis))
  allocate(xyz(data_size,natm,3))
  allocate(v(data_size),v_out(data_size),b(data_size),coeff(ncoeff),s(ncoeff))
  allocate(yij(data_size,ndis))
  allocate(p(ncoeff))
  allocate(A(data_size,ncoeff))
  allocate(wt(data_size))

  do i=1,data_size
     read(12,*)
     read(12,*) v(i)
     do j=1,natm
        read(12,*) symb,xyz(i,j,:)
     end do
  end do
  vmin = minval(v)

  do m=1,data_size
     k = 1
     do i=1,natm-1
        do j=i+1,natm
           yij(m,k)=0.0
           dr=xyz(m,i,:)-xyz(m,j,:)
           yij(m,k)=sqrt(dot_product(dr,dr))
           yij(m,k)=yij(m,k)/0.5291772083
           yij(m,k)=exp(-yij(m,k)/a0)

           k=k+1
        end do
     end do
  end do

  do i=1,data_size
     wt(i) = dwt / (dwt + v(i) - vmin)
     x     = yij(i,:)
     call bemsav(x, p)
     A(i,:) = p * wt(i)
     b(i) = v(i) * wt(i)
  end do

  call dgelss(data_size,ncoeff,1,A,data_size,b,data_size,s,1.0d-8,rank,work,150000,info)
  coeff(:)=b(1:ncoeff)

  do i=1,ncoeff
     write(10,*) coeff(i)
  end do
  
  HTOCM = 2.194746313702e5
 
  rmse  = 0.0
  wrmse = 0.0
  do i=1,data_size
     v_out(i) = emsav(yij(i,:), coeff)
     rmse     = rmse  + (v(i) - v_out(i))**2
     wrmse    = wrmse + (wt(i) * (v(i) - v_out(i)))**2
     write (11,*) v(i) * HTOCM, v_out(i) * HTOCM, abs(v(i) - v_out(i)) * HTOCM 
  end do

  rmse  = sqrt(rmse  / dble(data_size)) * HTOCM
  wrmse = sqrt(wrmse / dble(data_size)) * HTOCM
  write(*,'(A)') '3. Fitting is finished: '
  write(*,'(A,F15.10,A)') 'Overall  Root-mean-square fitting error: ', rmse,  ' cm-1'
  write(*,'(A,F15.10,A)') 'Weighted Root-mean-square fitting error: ', wrmse, ' cm-1'

  deallocate(xyz,x,v,b,p,yij,A,coeff,v_out,s,wt)
  close (10)
  close (11)
  close (12)
end program
"""


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    wdir         = "./H2-H2O"
    order        = "3"
    symmetry     = "2 2 1"
    config_fname = "points.dat"
    dataset = PolyDataset(wdir=wdir, config_fname=config_fname, order=order, symmetry=symmetry)

    print("NATOMS:   {}".format(dataset.NATOMS))
    print("NPOLY:    {}".format(dataset.NPOLY))
    print("NCONFIGS: {}".format(dataset.NCONFIGS))

    fname = os.path.join(wdir, "fit.f90")
    logging.info("generating Fortran code to perform least-squares fit: {}".format(fname))
    with open(fname, mode='w') as out:
        code = generate_fit_code(config_fname, dataset.NATOMS, dataset.NPOLY, dataset.NCONFIGS)
        out.write(code)

    logging.info("done.")

    logging.info("compiling fitting program...")
    basis = "basis_{}_{}".format(symmetry.replace(' ', '_'), order)
    basis = os.path.join(wdir, basis)
    cl("gfortran -fdefault-real-8 -c {0}.f90 -o {0}.o".format(basis))
    obj = os.path.join(wdir, "fit.o")
    cl("gfortran -fdefault-real-8 -c {} -o {}".format(fname, obj))
    exe = os.path.join(wdir, "fit.x")
    cl("gfortran {}.o {} -o {} -llapack".format(basis, obj, exe))
