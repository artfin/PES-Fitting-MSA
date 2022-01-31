# Jun Li, Bin Jiang, and Hua Guo
# J. Chem. Phys. 139, 204103 (2013); https://doi.org/10.1063/1.4832697
# Suggest using Tanh activation function and 2 hidden layers

import torch.nn as nn

class FCNet(nn.Module):
    def __init__(self, NPOLY, n_layers=2, activation=nn.Tanh, init_form="uniform"):
        super().__init__()

        self.NPOLY      = NPOLY
        self.n_layers   = n_layers
        self.activation = activation()
        self.init_form  = init_form

        layers = [
            nn.Linear(self.NPOLY, 20), self.activation,
            nn.Linear(20, 20),         self.activation,
            nn.Linear(20, 1)
        ]

        self.layers = nn.Sequential(*layers)

        if isinstance(self.activation, nn.ReLU):
            self.init_kaiming(activation_str="relu")
        elif isinstance(self.activation, nn.Tanh):
            self.init_xavier(activation_str="tanh")
        elif isinstance(self.activation, nn.Sigmoid):
            self.init_xavier(activation_str="sigmoid")
        else:
            raise NotImplementedError()

    def forward(self, x):
        return self.layers(x)

    def init_xavier(self, activation_str):
        sigmoid_gain = nn.init.calculate_gain(activation_str)
        for child in self.layers.children():
            if isinstance(child, nn.Linear):
                for _ in range(0, self.n_layers - 1):
                    if self.init_form == "normal":
                        nn.init.xavir_normal_(child.weight, gain=sigmoid_gain)
                        if child.bias is not None:
                            nn.init.zeros_(child.bias)
                    elif self.init_form == "uniform":
                        nn.init.xavier_uniform_(child.weight, gain=sigmoid_gain)
                        if child.bias is not None:
                            nn.init.zeros_(child.bias)

                    else:
                        raise NotImplementedError()

    def init_kaiming(self, activation_str):
        raise NotImplementedError()
