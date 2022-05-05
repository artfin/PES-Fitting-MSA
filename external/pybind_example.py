import ctypes as ct
from itertools import combinations
import numpy as np
from numpy.ctypeslib import ndpointer

def xyz_to_internal(x):
    """
    x:
       xC  = x[0]  yC  = x[1]  zC  = x[2]
       xH1 = x[3]  yH1 = x[4]  zH1 = x[5]
       xH2 = x[6]  yH2 = x[7]  zH2 = x[8]
       xH3 = x[9]  yH3 = x[10] zH3 = x[11]
       xH4 = x[12] yH4 = x[13] zH4 = x[14]
    q: [0] r1 -- d(C-H1)
       [1] r2 -- d(C-H2)
       [2] r3 -- d(C-H3)
       [3] r4 -- d(C-H4)
       [4] alpha12 -- ang(H1-C-H2)
       [5] alpha13 -- ang(H1-C-H3)
       [6] alpha14 -- ang(H1-C-H4)
       [7] alpha23 -- ang(H2-C-H3)
       [8] alpha24 -- ang(H2-C-H4)
       [9] alpha34 -- ang(H3-C-H4)
    """

    q = np.zeros((10, 1))
    q[0] = np.sqrt((x[3]-x[0])*(x[3]-x[0]) + (x[4]-x[1])*(x[4]-x[1]) + (x[5]-x[2])*(x[5]-x[2]));
    q[1] = np.sqrt((x[6]-x[0])*(x[6]-x[0]) + (x[7]-x[1])*(x[7]-x[1]) + (x[8]-x[2])*(x[8]-x[2]));
    q[2] = np.sqrt((x[9]-x[0])*(x[9]-x[0]) + (x[10]-x[1])*(x[10]-x[1]) + (x[11]-x[2])*(x[11]-x[2]));
    q[3] = np.sqrt((x[12]-x[0])*(x[12]-x[0]) + (x[13]-x[1])*(x[13]-x[1]) + (x[14]-x[2])*(x[14]-x[2]));

    k = 4
    for i in range(1, 4):
        for j in range(i + 1, 5):
            u = np.array([x[3*i], x[3*i + 1], x[3*i + 2]])
            u = u / np.linalg.norm(u)

            v = np.array([x[3*j], x[3*j + 1], x[3*j + 2]])
            v = v / np.linalg.norm(v)

            q[k] = np.arccos(np.dot(u, v)) / np.pi
            k = k + 1

    return q

class Poten_CH4:
    LIBPATH = "obj/xy4.so"

    def __init__(self, libpath=LIBPATH):
        basislib = ct.CDLL(libpath)

        potinit = basislib.potinit
        potinit.argtypes = []
        potinit()

        self.poten_xy4 = basislib.poten_xy4
        self.poten_xy4.argtypes = [ndpointer(ct.c_double, flags="F_CONTIGUOUS"), ndpointer(ct.c_double, flags="F_CONTIGUOUS")]


    def eval(self, x):
        """
        x: cartesian coordinates: C, H1, H2, H3, H4
        """
        q = xyz_to_internal(x)

        V = np.zeros((1, 1))
        self.poten_xy4(q, V)

        return V[0][0]

if __name__ == "__main__":

    pes = Poten_CH4()

    x = np.array([
         0.0000000000,  0.0000000000,  0.0000000000,
         1.0860100000,  0.0000000000,  0.0000000000,
        -0.3620033333,  0.0000000000, -1.0239000473,
        -0.3620033333, -0.8867234520,  0.5119500235,
        -0.3620033333,  0.8867234517,  0.5119500240,
    ])

    V = pes.eval(x)
    print("V: {}".format(V))
