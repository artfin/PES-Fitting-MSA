import logging

import os
import shlex
from subprocess import Popen, PIPE, STDOUT

import sys
if sys.version_info > (3, 0):
	raw_input = input

from collections import namedtuple
cmdstat = namedtuple('cmdstat', ['stdout', 'stderr', 'returncode'])

import pathlib
BASEDIR = pathlib.Path(__file__).parent.parent.resolve()

def cl(command):
    p = Popen(command, stdout=PIPE, shell=True)
    stdout, stderr = p.communicate()

    stdout     = stdout.decode("utf-8").strip()
    returncode = p.returncode

    try:
        stderr = stderr.decode("utf-8").strip()
    except:
        stderr = ""

    return cmdstat(stdout, stderr, returncode)

def build_lib():
    c_src = os.path.join(BASEDIR, "c_src")
    st = cl('cd {}; make msa'.format(c_src))

    if st.returncode == 0:
        logging.info(st.stdout)
    else:
        logging.error("return code = {}".format(st.returncode))
        logging.error("during the cmd = {}".format(st.stdout))
        sys.exit(1)

def run_msa(order, symmetry, wdir):
    exe = os.path.join(BASEDIR, "c_src", "msa")
    cmd = '{} {} {}'.format(exe, order, symmetry)
    logging.info("cmd: {}".format(cmd))
    st = cl(cmd)
    logging.info(st.stdout)

    if st.returncode == 0:
        logging.info("PIP basis files successfully generated.")
    else:
        logging.error("MSA return code = {}".format(st.returncode))
        sys.exit(1)

    fname = 'MOL_{}_{}'.format(symmetry.replace(' ', '_'), order)

    st = cl('mv -v {}.BAS  {}'.format(fname, wdir)); logging.info(st.stdout)
    st = cl('mv -v {}.FOC  {}'.format(fname, wdir)); logging.info(st.stdout)
    st = cl('mv -v {}.MAP  {}'.format(fname, wdir)); logging.info(st.stdout)
    st = cl('mv -v {}.MONO {}'.format(fname, wdir)); logging.info(st.stdout)
    st = cl('mv -v {}.POLY {}'.format(fname, wdir)); logging.info(st.stdout)

def compile_dlib(order, symmetry, wdir):
    logging.info("postmsa.pl generates Fortran code...")

    perl_script = os.path.join(BASEDIR, "src", "postmsa.pl")
    cl('perl {0} {1} {2} {3}'.format(perl_script, wdir, order, symmetry))

    logging.info("compiling dynamic lib...")
    fname = "basis_{}_{}".format(symmetry.replace(' ', '_'), order)
    fpath = os.path.join(wdir, fname)
    st = cl('gfortran -shared -fdefault-real-8 {0}.f90 -o {0}.so'.format(fpath))
    logging.info(st.stdout)


if __name__   == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logging.info("building MSA library...\n")
    build_lib()

    order         = raw_input('Please enter the maximum order of the polynomial: ')
    symmetry      = raw_input('Please enter the permutation symmetry of the molecule: ')
    wdir          = raw_input('Please enter the working directory: ')

    logging.info("generating PIP basis:\n")
    logging.info("    order        = {}".format(order))
    logging.info("    symmetry     = {}".format(symmetry))
    logging.info("    wdir         = {}".format(wdir))

    run_msa(order, symmetry, wdir)
    compile_dlib(order, symmetry, wdir)
    logging.info("Finished.")
