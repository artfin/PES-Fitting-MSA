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

curve = np.loadtxt("radial_curve.txt")

plt.figure(figsize=(10, 10))
ax = plt.subplot(1, 1, 1)

plt.plot(curve[:,0], curve[:,1], color='r', lw=2.0, label="NN-PIP PES")
plt.plot(curve[:,0], curve[:,2], color='k', lw=2.0, linestyle='--', label="Symmetry-adapted PES")

plt.legend(fontsize=14)

plt.xlim((9.0, 30.0))
plt.ylim((-5.0, 0.0))

#plt.yscale('log')

plt.xlabel(r"R, a0")
plt.ylabel(r"Energy, cm$^{-1}$")
plt.title("Global minimum crossection")

ax.xaxis.set_major_locator(plt.MultipleLocator(5.0))
ax.xaxis.set_minor_locator(plt.MultipleLocator(1.0))
ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))

ax.tick_params(axis='x', which='major', width=1.0, length=6.0)
ax.tick_params(axis='x', which='minor', width=0.5, length=3.0)
ax.tick_params(axis='y', which='major', width=1.0, length=6.0)
ax.tick_params(axis='y', which='minor', width=0.5, length=3.0)

#plt.savefig("SVC-comparison.png", format='png', dpi=300)
plt.show()

