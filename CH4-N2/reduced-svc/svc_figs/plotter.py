import json
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

from collections import namedtuple
Point = namedtuple('Point', ['temperature', 'value', 'error', 'source'])

def load(filename):
    with open(filename, mode = 'r') as inp:
        return json.load(inp)

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

if __name__ == "__main__":
    experiment = load("experiment.json") 
    points = [Point(float(key), value[0], value[1], value[2]) for key, value in experiment.items()]

    sources = list(set([point.source for point in points]))

    marker_types = ['o', 'v', 'd', 's', 'x', '*', 'D', '<', '>']

    color_types = [
        (0.4878721993550683, 0.8781100104134166, 0.12707576924737085),
        (0.9339271821258623, 0.6638475490254921, 0.06887576137302098),
        (0.9657992036125628, 0.32946939028123734, 0.2644476505326797),
        (0.4238473958870087, 0.1431881192236123, 0.7781312082054912),
        (0.4643551973073071, 0.3414745573243757, 0.6690002259424683),
        (0.5026916222676358, 0.4057280158345905, 0.3028720207300768),
        (0.20090386474261657, 0.45376862969036522, 0.7483150665203994),
        (0.16515125950336174, 0.3197757752373891, 0.1311422689206181)
    ]

    markers = {source : marker for source, marker in zip(sources, marker_types)}
    colors = {source : color for source, color in zip(sources, color_types)}

    refs = {
        "01-ababio"  : r"Ababio et al., 2001",
        "71-ng"      : r"Ng, 1971",
        "72-Roe-Phd" : r"Roe, 1972",
        "82-mar/tre" : r"Martin et al., 1982",
        "88-jae/aud" : r"Jaeschke et al., 1988",
        "89-did/zhd" : r"Didovicher et al., 1989",
        "91-lop/roz" : r"Lopatinskii et al., 1991",
        "61-mas/eak" : r"Mason et al., 1961",
    }

    nnpip_pes = np.loadtxt("nnpippes-svc.txt")
    symm_pes  = np.loadtxt("symmpes-svc.txt")

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)

    plt.plot(symm_pes[:,0], symm_pes[:,1], color='k', lw=2.0, label="Symmetry-adapted PES")
    plt.plot(nnpip_pes[:,0], nnpip_pes[:,1], color='r', lw=2.0, label="NN-PIP PES")

    legend_elements = []
    for point in points:
        marker = markers[point.source]
        color = colors[point.source]
        el = plt.scatter(point.temperature, point.value, color=lighten_color(color, 0.8),
                         marker=marker, edgecolors='k', lw=0.5, s=60, zorder=2)

        if point.source != "01-ababio":
            plt.errorbar(point.temperature, point.value, yerr = point.error, color='k',
                         capsize=4, elinewidth=0.5, markeredgewidth=1.0)

        if point.source not in legend_elements:
            el.set_label(refs[point.source])
            legend_elements.append(point.source)

    plt.legend(fontsize=11, fancybox=True, frameon=True, shadow=True)

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

    plt.savefig("SVC-comparison.png", format='png', dpi=300)
    plt.show()
