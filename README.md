# KrakeNN 

This software is designed to construct permutationally invariant polynomial neural networks (PIP-NN) [[1]](https://doi.org/10.1063/1.4817187) to fit intermolecular potential energy surfaces.
The PIP-NN method imposes permutation symmetry using a set of PIPs as input. The invariance of the model with respect to overall translations and rotations is guaranteed through the use of interatomic distances as arguments of PIPs [[2]](https://doi.org/10.1080/01442350903234923).

Code in the repo is built on top of the MSA software used to construct the set of PIPs. As an example, we consider the intermolecular energy surface for the CH$_4$-N$_2$ van der Waals complex. First, we demonstrate the accuracy and robustness of the PIP-NN model within the rigid-rotor approximation (see folder `models/rigid/`). To attest to the high quality of the constructed model, we calculate the temperature variation of the cross second virial coefficient. We found the perfect agreement with previously published calculations [[3]](https://doi.org/10.1039/D1CP02161C) and reasonable agreement with experimental data.

<p align="center">
  <img src="https://github.com/artfin/PES-Fitting-MSA/blob/master/models/rigid/best-model/silu-svc-comparison.png " width="400">
</p>

Next, we trained a model on the dataset of intermolecular energies obtained for vibrationally excited moieties -- up to 3,000 cm-1 for CH$_4$ and 1,000 for N$_2$ (see folder `models/nonrigid/`). The figure demonstrates the high quality of constructed model we will use further in spectroscopic and thermophysical applications.

<p align="center">
  <img src="https://github.com/artfin/PES-Fitting-MSA/blob/master/models/nonrigid/nr-best-model/silu-ratio-clipped-purify-ch4-overview.png" width="400">
</p>

### Getting code and requirements 
```
* git clone https://github.com/artfin/PES-Fitting-MSA.git
* cd PES-Fitting-MSA
* pip install -r requirements.txt
```

KrakeNN also requires
```
* GNU Compiler Collection (specifically, gfortran)
```

Code was tested on Linux system with `Python=3.8`, `gcc=9.4.0`. 

### How do I train the model of CH4-N2 PES using KrakeNN?


The preparation of PIPs, model architecture, and training hyperparameters are configured through the YAML file. Let us consider the model training on the values of PIPs computed from the dataset of energies obtained within rigid-rotor approximation (uncorrected energies from `datasets/raw/CH4-N2-RIGID.xyz` and asymptotic energies from `datasets/raw/CH4-N2-RIGID-LIMITS.xyz`). As an example, let us take `model/rigid/best-model/silu.yaml` configuration file (shown in the next section). 

Model training using this YAML file can be started as follows:  
```python3 src/train_model.py --model_folder=models/rigid/best-model/ --model_name=silu --log_name=test --chk_name=test```

Logging info is shown in the command line and duplicated to provided `log_name` in the `model_folder`. If no `log_name` is provided, log is piped to `model_name.log` file in the `model_folder`. The same logic is applied to general checkpoints through the keyword `chk_name`. The event log for loss values on the training and validation sets is written to a subfolder `model_name` within the folder `runs`. The training in this configuration took approximately 75 minutes on NVIDIA A100 Tensor Core GPU.

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
```

The `DATASET` block describes the pipeline of PIPs preparation:
* `ORDER`:                  maximum order of PIP [no default; available: 3, 4, 5]
* `SYMMETRY`:               permutational symmetry of the molecular system; see more below [default: 4 2 1; available: 4 2 1]
* `TYPE`:                   selects files with configurations & intermolecular energies to be fitted [no default; available: RIGID, NONRIGID, NONRIGID-CLIP]
* `INTRAMOLECULAR_TO_ZERO`: whether to set to zero interfragment Morse variables [default: False] 
* `PURIFY`:                 whether to purify the set of PIPs, see more below [default: False]


For now, only the complex CH$_4$-N$_2$ is explored. The permutational group for the complex can be represented as S4 x S2 x S1 or, in short, "4 2 1". Here we arrange the atoms in the following order: H H H H N N C. The keyword `TYPE: RIGID` triggers the code to parse configurations from the hardcoded list of files, which contain energies corresponding to equilibrium geometries of both moieties. Keywords `INTRAMOLECULAR_TO_ZERO` and `PURIFY` specify the subset of the complete basis set of PIPs that will be used for model training. Setting `INTRAMOLECULAR_TO_ZERO` to `True` allows us to use only those polynomials that contain interfragment coordinates. Through the keyword `PURIFY,` a user can trigger the purification of the basis set of PIPs (see, e.g., [review](https://doi.org/10.1146/annurev-physchem-050317-021139)). The latter is an essential step to improve the precision of the PES  in the long-range. In the purified basis set, we eliminate those polynomials that do not contain interfragment variables. As a result, we obtain a separable basis set, meaning that each polynomial is guaranteed to vanish in the long-range.   
The `MODEL` block defines the hyperparameters of a neural network which consists of fully connected layers.
* `HIDDEN_DIMS` : list that specifies the number of neurons within hidden layers of the model [no default]
* `ACTIVATION`  : non-linear transformation applied to the input of neuron [no default; available all activations within `torch.nn`]
* `BN`          : whether to add `torch.nn.BatchNorm1d` layer after each fully connected layer [default: False]
* `DROPOUT`     : adds `torch.nn.Dropout` layer after each fully connected layer that randomly zeros some of the elements with provided probability [default: 0.0]

Note that our experiments showed that batch normalization and regularization through dropping out nodes is detrimental to the model training. Those options are left for running experiments with other molecular systems. 

The `LOSS` block defines the hyperparameters of the loss function.
* `NAME`        : criterion to measure the error between elements of the input and target [no default; available: weighted MSE (`WMSE`) and weighted RMSE (`WRMSE`)]
* `WEIGHT_TYPE` : functional dependence of weights on intermolecular energy for loss function [no default]  
  * `BOLTZMANN` : exponential decay of weights 
    $$w_i = \exp \left( -\frac{E_i}{E_\textrm{ref}} \right)$$ 
    * `EREF` : decay parameter [no default; float]  
  * `PS` : weights in the form suggested by [Partridge and Schwenke](https://doi.org/10.1063/1.473987) 
    $$w_i = \frac{1}{E_i^w} \frac{\textrm{tanh} \left( - 6 \cdot 10^{-4} \left( E_i^w - E_\textrm{max} \right) \right) + 1}{2}, \quad E_i^w = \max \left( E_\textrm{max}, E_i \right)$$  
    * `EMAX` : parameter that defines the upper range of switch [no default; float]  
  * `RATIO` : hyperbolic decay of weights
    $$w_i = \frac{\Delta}{\Delta + E_i - \textrm{min}(E_i)}$$  
    * `DWT` : decay parameter [no default; float] 

The `TRAINING` block defines the algorithms and their parameters for model training. 

* `OPTIMIZER` : optimization algorithm  
  * `NAME` : optimizer class within `torch.optim` [no default; available `LBFGS`, `Adam`]
  * `LR`   : learning rate [default: 1.0 for `LBFGS` and 1.0e-3 for `Adam`]
  * `TOLERANCE_GRAD`, `TOLERANCE_CHANGE`, `MAX_ITER` : parameter propagation for `LBFGS`
  * `WEIGHT_DECAY` : parameter propagation for `Adam` 

We use `LBFGS` as the primary optimization algorithm; `Adam` was used only for pretraining without much success.

* `SCHEDULER` : reduce learning rate when a target metric has stopped improving
  * `NAME` : scheduler class within `torch.optim` [no default; available `ReduceLROnPlateau`]
  * `LR_REDUCE_GAMMA`, `PATIENCE`, `THRESHOLD`, `THRESHOLD_MODE`, `COOLDOWN`, `MIN_LR` : parameter propagation for `ReduceLROnPlateau`
 
* `EARLY_STOPPING` : a form of regularization based on the value of a loss on the validation set
  * `PATIENCE`  :  number of epochs with no improvement of loss after which training will be stopped [default: 1000]
  * `TOLERANCE` : absolute minimum change in loss to qualify as an improvement [default: 0.1] 

### How do I evaluate trained model?

Figures showing differences between fitted and calculated energies can be generated using the `src/eval_model.py` script through the following command:   
`python3 src/eval_model.py --model_folder=models/rigid/best-model/ --model_name=silu --chk_name=test --EMAX=2000.0 --learning_overview=True`

Here, we provided the keyword `EMAX` through which we can change the upper value for intermolecular energy, which sets the boundary for the overview of model errors on training, validation, and testing sets. The script supports the values `EMAX=2000.0` and `EMAX=10000.0`.   

You can compare your results with the model trained for over 5,000 epochs [approximately 75 minutes on NVIDIA A100 Tensor Core GPU] through the following command:   
`python3 src/eval_model.py --model_folder=models/rigid/best-model/ --model_name=silu --chk_name=silu --EMAX=2000.0 --learning_overview=True`

### Exporting model to C++ 

After a model prototype has been trained in Python, we would like to export it to C++ to use it in high-performance code to calculate thermophysical or spectroscopic properties. The script `src/export_model.py` exports the model via TorchScript using the tracing mechanism (`torch.jit.trace`):  
```python3 src/export_model.py --model_folder=models/rigid/best-model --model_name=silu --chk_name=test```

To use an exported model in the C++ environment, we opt to calculate the temperature variation of the cross second virial coefficient. The integration over translational degrees of freedom is performed utilizing Adaptive Monte Carlo method VEGAS implemented in [`hep-mc` package](https://github.com/cschwan/hep-mc). The folder `external/rigid-svc` contains relevant code [this code has not been tested in other environments].

We plan to try other exporting mechanisms because the TorchScript compiler while utilizing Just-In-Time (JIT) compilation and other fancy features, comes with massive overhead resulting in quite suboptimal performance.   
