<div align="center">
<img src="https://github.com/icecreamZjy/ECC-predicts-CCS/blob/main/LOGO.png" width=580 height=150>
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

1.生成分子3D构象。使用[rdkit.Chem.rdDistGeom.EmbedMultipleConfs](https://www.rdkit.org/docs/source/rdkit.Chem.rdDistGeom.html?highlight=embedmultipleconfs#rdkit.Chem.rdDistGeom.EmbedMultipleConfs),function to compute atomic coordinates in 3D using distance geometry.

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = -1
    ps.maxAttempts = 1
    ps.numThreads = 0
    ps.useRandomCoords = True
    re = AllChem.EmbedMultipleConfs(mol, numConfs = 1, params = ps)
2.保存相关参数。例如：adduct set, atoms set, Minimum value in atomic coordinates, Maximum value in atomic coordinates
3. 

    DataSet = MyDataset(features, adj, edge_features, ccs)
    
*Optionnal args *
- features : Node feature matrix
- adj : Adjacency matrix
- edge_features : Edge feature matrix
- ccs : CCS of molecules

## 模型训练
根据自己的训练数据集，使用训练模型

## CCS预测
将Graph和Adduct输入已经训练好的SiGMA模型中，即可得到该分子的CCS预测值。


