<div align="center">
<img src="https://github.com/icecreamZjy/ECC-predicts-CCS/blob/main/LOGO.png" width=550 height=150>
</div>

## SiGMA

这是论文 *Ion Mobility Collision Cross Section Prediction Using **S**tructure **I**ncluded **G**raph **M**erged with **A**dduct.* 的代码库。   
其中包括：
- ECC.py
- GraphData.py
- model.py
### Package required: 
We recommend to use [conda](https://conda.io/docs/user-guide/install/download.html) and pip.
- [python3](https://www.python.org/) 3.7.7
- [rdkit](https://rdkit.org/) 2020.09.5     
- [tensorflow](https://www.tensorflow.org) 2.4.0
- [spektral](https://graphneural.network/) 1.0.5

## 数据预处理
SiGMA是基于图神经网络预测CCS的模型，所以我们需要将SMILES字符串转化为Graph。 相关方法见GraphData.py

## CCS预测
