from GraphData import *
from model import *
import pandas as pd
from pandas import Series,DataFrame

def Model_prediction(ifile,ParameterPath,mfileh5,ofile,Isevaluate = 0):
    '''
    * Predict
    *
    * Attributes
    * ----------
    * ifile         : File path for storing the data of smiles and adduct
    * ParameterPath : File path for storing model parameter data
    * mfileh5       : File path where the model is storeds
    * ofile         : Path to save ccs prediction values
    * Isevaluate    : Evaluate ?
    '''
    smiles, adduct, ccs = read_data(ifile)
#     XIBA = 100
#     smiles, adduct, ccs = smiles[:XIBA], adduct[:XIBA], ccs[:XIBA]
    print('## Read data : ',len(smiles))
    
    # Read the parameter file needed for prediction
    param = parameter.Parameter()
    with open(ParameterPath,'rb') as file:
        param  = pickle.loads(file.read())
        
    print('## All Atoms  : ', param.All_Atoms)
    print('## All Adduct : ', param.adduct_SET)
        
    smiles, adduct, ccs, Coordinate = Generating_coordinates(smiles, adduct, ccs, param.All_Atoms)
    print(len(smiles),smiles[0],adduct[0],ccs[0])
    print('## 3D coordinates generated successfully ')
    
    for i in range(len(Coordinate)):
        Coordinate[i] = (np.array(Coordinate[i]) - param.Min_Coor) / (param.Max_Coor - param.Min_Coor)
    
    adj, features, edge_features = convertToGraph(smiles, Coordinate, param.All_Atoms)
    DataSet = MyDataset(features, adj, edge_features, ccs)
    print('## Graph & Adduct dataset completed')
    
    ECC_Model = load_Model_from_file(mfileh5)
    print('## Model loading completed')
    
    re = predict(ECC_Model,param.adduct_SET,DataSet,adduct,)
    data = {'SMILES' : smiles,
            'Adduct' : adduct,
            'Ture CCS': ccs,
            'Predicted CCS':re}
    df = DataFrame(data)
    df.to_csv(ofile,index=False)
    print('## CCS predicted completed')
    
    if Isevaluate == 1:
        Bais(ccs,re)
    return Bais(ccs,re)

def Model_train(ifile, ParameterPath, ofile, ofileDataPath, EPOCHS, BATCHS, Vis, All_Atoms=[], adduct_SET=[]):
    '''
    * Train
    *
    * Attributes
    * ----------
    * ifile         : File path for storing the data of smiles and adduct
    * ParameterPath : Save path of related data parameters
    * ofileDataPath : File path for storing model parameter data
    * ofile         : File path where the model is stored
    '''
    # Read the smiles adduct CCS in the file
    smiles, adduct, ccs = read_data(ifile) 
    XIBA = 100
    smiles, adduct, ccs = smiles[:XIBA], adduct[:XIBA], ccs[:XIBA]
    print('## 读取了数据数量 : ',len(smiles))
    
    # 如果用户没有输入 元素个数，那么默认就是训练集中所有的元素集合s
    if len(All_Atoms) == 0:
        All_Atoms = GetSmilesAtomSet(smiles) # 计算训练集中使用的元素集合
    
    # 给输入的SMILES进行3d构象
    smiles, adduct, ccs, Coordinate = Generating_coordinates(smiles, adduct, ccs, All_Atoms)
    
    # 生成的坐标数据进行数据归一化处理
    ALL_DATA = []
    for i in Coordinate:
        for ii in i:
            ALL_DATA.append(ii[0]);ALL_DATA.append(ii[1]);ALL_DATA.append(ii[2]);   
    Max_Coor, Min_Coor = np.max(ALL_DATA), np.min(ALL_DATA)
    
    for i in range(len(Coordinate)):
        Coordinate[i] = (np.array(Coordinate[i]) - Min_Coor) / (Max_Coor - Min_Coor)
    # Adduct的种类集合
    if len(adduct_SET) == 0:
        adduct_SET = list(set(list(adduct)))
        adduct_SET.sort()
    
    print('## All element types : ', All_Atoms)
    print('## All adduct types : ', adduct_SET)
    
    # 将参数存储在对象中
    rw = Parameter(adduct_SET, All_Atoms, Max_Coor, Min_Coor)
    output_hal = open(ParameterPath, 'wb')
    output_hal.write(pickle.dumps(rw))
    output_hal.close()
    
    # 将输入的数据构造成Graph
    adj, features, edge_features = convertToGraph(smiles, Coordinate, All_Atoms)
    DataSet = MyDataset(features, adj, edge_features, ccs)
    print('## Build graph data successfully')
    print('## Dataset:',DataSet,' Node features:',DataSet[0].x.shape)
    
    np.save(ofileDataPath+"features.npy",features)
    np.save(ofileDataPath+"adj.npy",adj)
    np.save(ofileDataPath+"edge_features.npy",edge_features)
    np.save(ofileDataPath+"ccs.npy",ccs)
    np.save(ofileDataPath+"Adduct.npy",adduct)
    # 生产用于训练的模型
    ECC_Model = Mymodel(DataSet)
    # 训练模型
    ECC_Model = train(ECC_Model,DataSet,adduct,adduct_SET, EPOCHS = EPOCHS, BATCHS = BATCHS, Vis = Vis)
    # 保存模型
    ECC_Model.save(ofile)
    
    return ECC_Model