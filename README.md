# KrakeNN 

This software is designed to construct permutation invariant polynomial neural networks (PIP-NN) [1] to fit intermolecular potential energy surfaces.
The PIP-NN method imposes permutation symmetry using a set of PIPs as input. The invariance of the model with respect to overall translations and rotations is guaranteed through the use of interatomic distances as arguments of PIPs [2].

Code in the repo is built on top of the MSA software used to construct the set of PIPs. As an example, we consider the intermolecular energy surface for the CH4-N2 van der Waals complex. First, we demonstrate the accuracy and robustness of the PIP-NN model within the rigid-rotor approximation (see models/rigid/). To attest to the high quality of the constructed model, we calculate the temperature variation of the cross second virial coefficient. We found the perfect agreement with previously published calculations [3] and reasonable agreement with experimental data.

Next, we trained a model on the dataset of intermolecular energies obtained for vibrationally excited moieties -- up to 3,000 cm-1 for CH4 and 1,000 for N2 (see models/nonrigid/). The figure demonstrates the high quality of constructed model that can be further utilized in spectroscopic and thermophysical applications.

[1] Jiang B., Guo H. Permutation invariant polynomial neural network approach to fitting potential energy surfaces // Journal of Chemical Physics. 2013. Vol. 139, no. 5. P. 054112–054112    
[2] Braams B. J., Bowman J. M. Permutationally invariant potential energy surfaces in high dimensionality // International Reviews in Physical Chemistry. 2009. Vol. 28, no. 4. P. 577–606   
[3] Finenko A. A., Chistikov D. N., Kalugina Y. N. et al. Fitting potential energy and induced dipole surfaces of the van der Waals complex CH4–N2 using non-product quadrature grids // Physical Chemistry Chemical Physics (Incorporating Faraday Transactions). 2021. Vol. 23, no. 34. P. 18475–18494   

### How do I train the model of CH4-N2 PES using KrakeNN?

* Make sure that all the required packages are available in your command line environment on Linux  
    * `git clone https://github.com/artfin/PES-Fitting-MSA.git`  
    * `cd PES-Fitting-MSA`  
    * `pip install -r requirements.txt`   

* The preparation of PIPs, model architecture and training hyperparameters are configured through the YAML file. Let us consider the training of model on the values of PIPs computed from the dataset of energies obtained within rigid-rotor approximation (uncorrected energies from `datasets/raw/CH4-N2-RIGID.xyz` and asymptotic energies from `datasets/raw/CH4-N2-RIGID-LIMITS.xyz`). As an example, let us take [``model/rigid/best-model/silu.yaml```](model/rigid/best-model/silu.yaml) configuration file.  

Training of the model using this YAML file can be started as follows:  
```python3 src/train_model.py --model_folder=models/rigid/best-model/ --model_name=silu --log_path=test.log```


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
* `SYMMETRY`:               permutational symmetry of the molecular system, see more below [default: 4 2 1, available: 4 2 1]
* `TYPE`:                   selects files with configurations & intermolecular energies to be fitted [no default; available: RIGID, NONRIGID, NONRIGID-CLIP]
* `INTRAMOLECULAR_TO_ZERO`: whether to set to zero Morse variables corresponding to interatomic distance for which both of the atoms belong to one moiety [default: False] 
* `PURIFY`:                 whether to purify the set of PIPs, see more below [default: False]

For now, only the complex CH4-N2 is explored, so the permutational group described by the string "4 2 1" is expected 
 The keyword `TYPE: RIGID` triggers the code to parse configurations obtained within rigid-rotor approximation (using the filepaths hardcoded in the application). Then the      
 


### Pipeline

* train model using `train_model.py` script
* plot errors using `eval_model.py` script
* export model to TorchScript using `export_model.py` script 

