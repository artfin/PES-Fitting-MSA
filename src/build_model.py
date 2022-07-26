import collections
import logging

import torch

class Builder:
    def __init__(self, *namespaces):
        self._namespace = collections.ChainMap(*namespaces)

    def __call__(self, name, *args, **kwargs):
        try:
            return self._namespace[name](*args, **kwargs)
        except Exception as e:
            raise e.__class__(str(e), name, args, kwargs) from e

def build_network_yaml(architecture, input_features, output_features, builder=Builder(torch.nn.__dict__)):
    layers = []

    hidden_dims = architecture['HIDDEN_DIMS']
    activation  = architecture['ACTIVATION']
    bias        = architecture.get('BIAS', True)
    BN          = architecture.get('BN', False)
    dropout     = architecture.get('DROPOUT', 0.0)

    in_features = input_features
    out_features = input_features

    for i in range(len(hidden_dims)):
        out_features = hidden_dims[i]
        layers.append(torch.nn.Linear(in_features, out_features, bias=bias))
        if BN:
            layers.append(torch.nn.BatchNorm1d(out_features))
        if dropout > 0:
            layers.append(torch.nn.Dropout(p=dropout))

        layers.append(builder(activation))
        in_features = out_features

    layers.append(torch.nn.Linear(out_features, output_features, bias=bias))

    model = torch.nn.Sequential(*layers)
    model.double()

    logging.info("Build model: {}".format(model))

    return model
