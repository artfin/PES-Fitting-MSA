import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] =True
plt.rcParams["mathtext.fontset"] = "cm"

mpl.rcParams['font.serif'] = 'Times'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['axes.titlesize'] = 15
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15

nnpip_pes = np.loadtxt("nnpippes-svc.txt")
symm_pes  = np.loadtxt("symmpes-svc.txt")

plt.figure(figsize=(10, 10))
ax = plt.subplot(1, 1, 1)

plt.plot(symm_pes[:,0], symm_pes[:,1], color='k', lw=2.0, label="Symmetry-adapted PES")
plt.plot(nnpip_pes[:,0], nnpip_pes[:,1], color='r', lw=2.0, label="NN-PIP PES")

plt.legend(fontsize=14)

plt.xlim((150.0, 500.0))
plt.ylim((-140.0, 30.0))

plt.xlabel(r"Temperature, K")
plt.ylabel(r"$B_{12}$, cm$^3 \cdot$mol$^{-1}$")

ax.xaxis.set_major_locator(plt.MultipleLocator(50.0))
ax.xaxis.set_minor_locator(plt.MultipleLocator(10.0))
ax.yaxis.set_major_locator(plt.MultipleLocator(20.0))
ax.yaxis.set_minor_locator(plt.MultipleLocator(10.0)) 

ax.tick_params(axis='x', which='major', width=1.0, length=6.0)
ax.tick_params(axis='x', which='minor', width=0.5, length=3.0)
ax.tick_params(axis='y', which='major', width=1.0, length=6.0)
ax.tick_params(axis='y', which='minor', width=0.5, length=3.0)

#plt.savefig("SVC-comparison.png", format='png', dpi=300)
plt.show()
