import collections
import logging
import itertools
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Builder:
    def __init__(self, *namespaces):
        self._namespace = collections.ChainMap(*namespaces)

    def __call__(self, name, *args, **kwargs):
        try:
            return self._namespace[name](*args, **kwargs)
        except Exception as e:
            raise e.__class__(str(e), name, args, kwargs) from e


def build_network(cfg_model, hidden_dims, input_features, output_features, builder=Builder(torch.nn.__dict__)):
    layers = []

    #hidden_dims = architecture['HIDDEN_DIMS']
    activation  = cfg_model['ACTIVATION']
    bias        = cfg_model.get('BIAS', True)
    BN          = cfg_model.get('BN', False)
    dropout     = cfg_model.get('DROPOUT', 0.0)

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

    return model

class QModel(torch.nn.Module):
    def __init__(self, cfg_model, input_features, output_features):
        super(QModel, self).__init__()

        self.input_features = input_features
        self.input_features_acc = list(itertools.accumulate(self.input_features))

        hidden_dims = cfg_model['HIDDEN_DIMS']
        assert len(input_features) == len(output_features), "input_features: {}; output_features: {}".format(input_features, output_features)
        assert len(hidden_dims) == len(input_features), "hidden_dims: {}; input_features: {}".format(hidden_dims, input_features)

        self.blocks = torch.nn.ModuleList([
            build_network(cfg_model, hd, i, o)
            for hd, i, o in zip(hidden_dims, input_features, output_features)
        ])


    def forward(self, X):
        out = torch.empty(0).to(DEVICE)
        for block, i1, i2 in zip(self.blocks, [0, *self.input_features_acc], [*self.input_features_acc, sum(self.input_features_acc)]):
            out = torch.cat((out, block(X[:, i1:i2])), dim=1)

        return out 
