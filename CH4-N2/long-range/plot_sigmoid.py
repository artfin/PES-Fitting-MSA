import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    C = 15.0 # sigmoid center of symmetry
    S = 0.5 # the speed of `turning on` the long-range model 

    R = np.linspace(4.5, 30.0, 100)
    WT = 1.0 / (1.0 + np.exp(-S * (R - C)))

    plt.figure(figsize=(12, 8))

    plt.plot(R, WT, color='r')

    plt.show()
