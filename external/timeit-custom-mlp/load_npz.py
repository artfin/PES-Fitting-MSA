#!/usr/bin/env python

import sys
from numpy import load

if __name__ == "__main__":
    data = load("silu.npz")

    for key in data.keys():
        print(key)

    mean = data["xscaler.mean"]
    scale = data["xscaler.scale"]

    print(data["2.bias"].shape)

