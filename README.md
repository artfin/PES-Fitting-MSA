# KrakeNN 

This software is designed to construct permutation invariant polynomial neural networks (PIP-NN) [[1]](https://doi.org/10.1063/1.4817187) to fit intermolecular potential energy surfaces.
The PIP-NN method imposes permutation symmetry using a set of PIPs as input. The invariance of the model with respect to overall translations and rotations is guaranteed through the use of interatomic distances as arguments of PIPs [[2]](https://doi.org/10.1080/01442350903234923).

Code in the repo is built on top of the MSA software used to construct the set of PIPs. As an example, we consider the intermolecular energy surface for the CH4-N2 van der Waals complex. First, we demonstrate the accuracy and robustness of the PIP-NN model within the rigid-rotor approximation (see folder `models/rigid/`). To attest to the high quality of the constructed model, we calculate the temperature variation of the cross second virial coefficient. We found the perfect agreement with previously published calculations [[3]](https://doi.org/10.1039/D1CP02161C) and reasonable agreement with experimental data.

<p align="center">
  <img src="https://github.com/artfin/PES-Fitting-MSA/blob/master/models/rigid/best-model/silu-svc-comparison.png " width="400">
</p>

Next, we trained a model on the dataset of intermolecular energies obtained for vibrationally excited moieties -- up to 3,000 cm-1 for CH4 and 1,000 for N2 (see folder `models/nonrigid/`). The figure demonstrates the high quality of constructed model we will use further in spectroscopic and thermophysical applications.

<p align="center">
  <img src="https://github.com/artfin/PES-Fitting-MSA/blob/master/models/nonrigid/nr-best-model/silu-ratio-clipped-purify-ch4-overview.png" width="400">
</p>

### How do I train the model of CH4-N2 PES using KrakeNN?

Make sure all the required packages are available in your command line environment on Linux  
    * `git clone https://github.com/artfin/PES-Fitting-MSA.git`  
    * `cd PES-Fitting-MSA`  
    * `pip install -r requirements.txt`   

The preparation of PIPs, model architecture, and training hyperparameters are configured through the YAML file. Let us consider the model training on the values of PIPs computed from the dataset of energies obtained within rigid-rotor approximation (uncorrected energies from `datasets/raw/CH4-N2-RIGID.xyz` and asymptotic energies from `datasets/raw/CH4-N2-RIGID-LIMITS.xyz`). As an example, let us take `model/rigid/best-model/silu.yaml` configuration file.  

Model training using this YAML file can be started as follows:  
```python3 src/train_model.py --model_folder=models/rigid/best-model/ --model_name=silu --log_path=test.log```

Logging info is shown in the command line and duplicated to provided `log_path`. If no `log_path` is provided, the logging file is created in the folder containing the configuration file. Checkpoints during training are saved to the `model_folder` as `checkpoint.pt`, and the final checkpoint is renamed `model_name.pt`. The event log for loss values on the training and validation sets is written to a subfolder `model_name` within the folder `runs`.    


### What is the structure of YAML configuration file?

``` yaml
# models/rigid/best-model/silu.yaml
DATASET: 
  ORDER: 4
  SYMMETRY: 4 2 1
  TYPE: RIGID
  INTRAMOLECULAR_TO_ZERO: True
  NORMALIZE: std 

MODEL:
  ACTIVATION: SiLU
  HIDDEN_DIMS: [32]

LOSS:
  NAME: WMSE
  WEIGHT_TYPE: Ratio 
  dwt: 1000.0

TRAINING:
  MAX_EPOCHS: 10000
  OPTIMIZER:
    NAME: LBFGS
    LR: 1.0
  SCHEDULER:
    NAME: ReduceLROnPlateau
    LR_REDUCE_GAMMA: 0.8
    PATIENCE: 200 
    THRESHOLD: 0.1
    THRESHOLD_MODE: abs
    COOLDOWN: 3
    MIN_LR: 1.0e-5
  EARLY_STOPPING:
    PATIENCE: 1000 
    TOLERANCE: 0.02

OUTPUT_PATH: models/rigid/best-model
```

The `DATASET` block describes the pipeline of PIPs preparation:
* `ORDER`:                  maximum order of PIP [no default; available: 3, 4, 5]
* `SYMMETRY`:               permutational symmetry of the molecular system; see more below [default: 4 2 1; available: 4 2 1]
* `TYPE`:                   selects files with configurations & intermolecular energies to be fitted [no default; available: RIGID, NONRIGID, NONRIGID-CLIP]
* `INTRAMOLECULAR_TO_ZERO`: whether to set to zero interfragment Morse variables [default: False] 
* `PURIFY`:                 whether to purify the set of PIPs, see more below [default: False]


For now, only the complex CH4-N2 is explored. The permutational group for the complex can be represented as S4 x S2 x S1 or, in short, "4 2 1". Here we arrange the atoms in the following order: H H H H N N C. The keyword `TYPE: RIGID` triggers the code to parse configurations from the hardcoded list of files, which contain energies corresponding to equilibrium geometries of both moieties. Keywords `INTRAMOLECULAR_TO_ZERO` and `PURIFY` specify the subset of the complete basis set of PIPs that will be used for model training. Setting `INTRAMOLECULAR_TO_ZERO` to `True` allows us to use only those polynomials that contain interfragment coordinates. Through the keyword `PURIFY,` a user can trigger the purification of the basis set of PIPs (see, e.g., [review](https://doi.org/10.1146/annurev-physchem-050317-021139)). The latter is an essential step to improve the precision of the PES  in the long-range. In the purified basis set, we eliminate those polynomials that do not contain interfragment variables. As a result, we obtain a separable basis set, meaning that each polynomial is guaranteed to vanish in the long-range.   
The `MODEL` block defines the hyperparameters of a neural network which consists of fully connected layers.
* `HIDDEN_DIMS` : list that specifies the number of neurons within hidden layers of the model [no default]
* `ACTIVATION`  : non-linear transformation applied to the input of neuron [no default; available all activations within `torch.nn`]
* `BN`          : whether to add `BatchNorm1d` layer after each fully connected layer [default: False]
* `DROPOUT`     : adds `torch.nn.Dropout` layer after each fully connected layer that randomly zeros some of the elements with provided probability [default: 0.0]

Experiments showed that batch normalization and regularization through dropping out nodes is detrimental to the model training. Those options are left for running experiments with other molecular systems. 

The `LOSS` block defines the hyperparameters of the loss function.
* `NAME`        : criterion to measure the error between elements of the input and target [no default; available: weighted MSE (`WMSE`) and weighted RMSE (`WRMSE`)]
* `WEIGHT_TYPE` : functional dependence of weights on intermolecular energy for loss function [no default]  
   * `Boltzmann` : exponential decay of weights 
      $$w_i = \exp \left( -\frac{E_i}{E_\textrm{ref}} \right)$$ 
      * `EREF` : decay parameter [no default; float]  
   * `PS` : weights in the form suggested by [Partridge and Schwenke](https://doi.org/10.1063/1.473987) 
      $$w_i = \frac{1}{E_i^w} \frac{\textrm{tanh} \lb - 6 \cdot 10^{-4} \lb E_i^w - E_\textrm{max} \rb \rb + 1}{2}, \quad E_i^w = \max \left( E_\textrm{max}, E_i \right)$$
