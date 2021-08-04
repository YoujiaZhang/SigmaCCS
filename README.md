## SiGMA
This is the repository of codes for the paper entitled "[Predicting Molecular Fingerprint from Electron-Ionization Mass Spectrum with Deep Neural Networks](https://pubs.acs.org/doi/10.1021/acs.analchem.0c01450)". This repository only contains the source codes without any data or pretrained models, due to the models were trained by NIST dataset.

<div align="center">
<img src="https://github.com/hcji/DeepEI/blob/master/figure.png">
</div>

### Depends:
[python3](https://www.python.org/)     
[rdkit](https://rdkit.org/)     
[tensorflow](https://www.tensorflow.org)     

optinal:    
[pycdk](https://github.com/hcji/pycdk)      
[smiles_to_onehot](https://gitee.com/hcji/smiles_to_onehot)    

### Data preprocess

Data preprocess scripts are used for extracting compound information of NIST into numpy object. They are included in the *scripts/read.py* , including gathering the SMILES, exact masses, retention indices, Morgan fingerprints, molecular descriptors and mass spectra.

### Training the model

DeepEI contain two main parts of models: 1. Predicting molecular fingerprint from EI-MS (*Fingerprint* folder); 2. Predicting retention index from structure (*retention* folder). Each folder contains the codes for data pretreatment, model training and model selection. For FP prediction, we compared for models, which are MLP, XGBoost, LR and PLS-DA. For RI prediction, we compared single-channel CNN, multi-channel CNN and MLP.

### Prediction

The main functions of predication are included in the *DeepEI* folder. *predict_RI* function takes SMILES as input and predicts the corresponding retention index. *predict_fingerprint* function takes mass spectrum as input and predicts the corresponding fingerprints. 

### Comparison

The *Discussion* folder contains the scripts for evaluating the identification performance, and comparing with [NEIMS](https://github.com/brain-research/deep-molecular-massspec) package. The corresponding results are also included. We compared DeepEI, NEIMS and their combination.

### Usage

The example codes for usage is included in the *Usage.ipynb*

**Contact:** ji.hongchao@foxmail.com
