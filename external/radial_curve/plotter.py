from collections import namedtuple
from subprocess import Popen, PIPE, STDOUT

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] =True
plt.rcParams["mathtext.fontset"] = "cm"

mpl.rcParams['font.serif'] = 'Times'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['axes.labelsize'] = 21
mpl.rcParams['axes.titlesize'] = 21
mpl.rcParams['xtick.labelsize'] = 21
mpl.rcParams['ytick.labelsize'] = 21

curve = np.loadtxt("radial-curve.txt")

def plot_long_range(figname=None):
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)

    plt.plot(curve[:,0], curve[:,1], color='#FF6F61', lw=2.0, label="NN-PIP")
    plt.plot(curve[:,0], curve[:,2], color='#CFBFF7', lw=2.0, linestyle='--', label="Symmetry-adapted angular basis")

    plt.legend(fontsize=18)

    plt.xlim((9.0, 30.0))
    plt.ylim((-5.0, 0.0))

    plt.xlabel(r"R, a0")
    plt.ylabel(r"Energy, cm$^{-1}$")
    plt.title("Global minimum crossection")

    ax.xaxis.set_major_locator(plt.MultipleLocator(5.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

    ax.tick_params(axis='x', which='major', width=1.0, length=6.0)
    ax.tick_params(axis='x', which='minor', width=0.5, length=3.0)
    ax.tick_params(axis='y', which='major', width=1.0, length=6.0)
    ax.tick_params(axis='y', which='minor', width=0.5, length=3.0)

    if figname is not None:
        plt.savefig(figname, format='png', dpi=300)

    plt.show()

def plot_potential_well(figname=None):
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)

    plt.plot(curve[:,0], curve[:,1], color='#FF6F61', lw=2.0, label="NN-PIP")
    plt.plot(curve[:,0], curve[:,2], color='#CFBFF7', lw=2.0, linestyle='--', label="Symmetry-adapted angular basis")

    plt.legend(fontsize=18)

    plt.xlim((5.5, 9.0))
    plt.ylim((-180.0, 0.0))

    plt.xlabel(r"R, a0")
    plt.ylabel(r"Energy, cm$^{-1}$")
    plt.title("Global minimum crossection")

    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(50.0))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(10.0))

    ax.tick_params(axis='x', which='major', width=1.0, length=6.0)
    ax.tick_params(axis='x', which='minor', width=0.5, length=3.0)
    ax.tick_params(axis='y', which='major', width=1.0, length=6.0)
    ax.tick_params(axis='y', which='minor', width=0.5, length=3.0)

    if figname is not None:
        plt.savefig(figname, format='png', dpi=300)

    plt.show()

cmdstat = namedtuple('cmdstat', ['stdout', 'stderr', 'returncode'])
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


def trim_png(figname):
    cl('convert {0} -trim +repage {0}'.format(figname))

if __name__ == "__main__":
    #figname = "potential-well.png"
    #plot_potential_well(figname=figname)
    #trim_png(figname)

    #figname = "long-range.png"
    #plot_long_range(figname=figname)
    #trim_png(figname)

    #plot_potential_well()
    plot_long_range()

