<div align="center">

<img src="https://github.com/icecreamZjy/ECC-predicts-CCS/blob/main/LOGO.png" width=580 height=150>    
<img src="https://github.com/icecreamZjy/ECC-predicts-CCS/blob/main/LOGO1.png" width=150 height=150>
</div>

## SigmaCCS

这是论文 *Ion Mobility Collision Cross Section Prediction Using **S**tructure **I**ncluded **G**raph **M**erged with **A**dduct.* 的代码库。   
其中包括：
- sigma.py
- GraphData.py
- model.py
- *data Folder*:  
    - TrainData.csv
    - TestData.csv
    - TestData-pred.csv (Predicted results)
- *model Folder*:
    - model.h5
- *parameter Folder*:
    - parameter.pkl 
### Package required: 
We recommend to use [conda](https://conda.io/docs/user-guide/install/download.html) and [pip](https://pypi.org/project/pip/).
- [python3](https://www.python.org/) 3.7.7
- [rdkit](https://rdkit.org/) 2020.09.5     
- [tensorflow](https://www.tensorflow.org) 2.4.0
- [spektral](https://graphneural.network/) 1.0.5

## 数据预处理
SiGMA是基于图神经网络预测CCS的模型，所以我们需要将SMILES字符串转化为Graph。 相关方法见`GraphData.py`      

**1.** 生成分子3D构象.   

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = -1
    ps.maxAttempts = 1
    ps.numThreads = 0
    ps.useRandomCoords = True
    re = AllChem.EmbedMultipleConfs(mol, numConfs = 1, params = ps)
    re = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads = 0)
- [ETKDGv3](https://www.rdkit.org/docs/source/rdkit.Chem.rdDistGeom.html?highlight=etkdgv3#rdkit.Chem.rdDistGeom.ETKDGv3) Returns an EmbedParameters object for the ETKDG method - version 3 (macrocycles).
- [EmbedMultipleConfs](https://www.rdkit.org/docs/source/rdkit.Chem.rdDistGeom.html?highlight=embedmultipleconfs#rdkit.Chem.rdDistGeom.EmbedMultipleConfs), use distance geometry to obtain multiple sets of coordinates for a molecule.
- [MMFFOptimizeMoleculeConfs](https://www.rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html?highlight=mmffoptimizemoleculeconfs#rdkit.Chem.rdForceFieldHelpers.MMFFOptimizeMoleculeConfs), uses MMFF to optimize all of a molecule’s conformations      

**2.** 保存相关参数.For details, see`parameter.py`.例如:    
- adduct set  
- atoms set   
- Minimum value in atomic coordinates   
- Maximum value in atomic coordinates   

**3.** 生成Graph数据集. 生成用于构造Graph的三个矩阵:(1) *node feature matrix*, (2) *adjacency matrix*, (3) *edge feature matrix*.    

    adj, features, edge_features = convertToGraph(smiles, Coordinate, All_Atoms)
    DataSet = MyDataset(features, adj, edge_features, ccs)
*Optionnal args*
- All_Atoms : The set of all elements in the dataset
- Coordinate : Array of coordinates of all molecules
- features : Node feature matrix
- adj : Adjacency matrix
- edge_features : Edge feature matrix

## 模型训练
根据自己的训练数据集，训练模型。

    Model_train(ifile, ParameterPath, ofile, ofileDataPath, EPOCHS, BATCHS, Vis, All_Atoms=[], adduct_SET=[])
*Optionnal args*
- ifile : File path for storing the data of smiles and adduct.
- ofile : File path where the model is stored.
- ParameterPath : Save path of related data parameters.
- ofileDataPath : File path for storing model parameter data.

## CCS预测
将Graph和Adduct输入已经训练好的SiGMA模型中，即可得到该分子的CCS预测值。

    Model_prediction(ifile, ParameterPath, mfileh5, ofile, Isevaluate = 0)
*Optionnal args*
- ifile : File path for storing the data of smiles and adduct
- ParameterPath : File path for storing model parameter data
- mfileh5 : File path where the model is stored
- ofile : Path to save ccs prediction values

## Usage
The example codes for usage is included in the [test.ipynb](main/test.ipynb)
