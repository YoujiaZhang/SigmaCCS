from GraphData import *
from model import *
import pandas as pd
from pandas import Series,DataFrame
from parameter import *

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
    # print('## Read data : ',len(smiles))
    param = parameter.Parameter()
    with open(ParameterPath,'rb') as file:
        param  = pickle.loads(file.read())
    # print('## All Atoms  : ', param.All_Atoms)
    # print('## All Adduct : ', param.adduct_SET)
    smiles, adduct, ccs, Coordinate = Generating_coordinates(smiles, adduct, ccs, param.All_Atoms)
    # print(len(smiles),smiles[0],adduct[0],ccs[0])
    # print('## 3D coordinates generated successfully ')
    
    for i in range(len(Coordinate)):
        Coordinate[i] = (np.array(Coordinate[i]) - param.Min_Coor) / (param.Max_Coor - param.Min_Coor)
    
    adj, features, edge_features = convertToGraph(smiles, Coordinate, param.All_Atoms)
    DataSet = MyDataset(features, adj, edge_features, ccs)
    # print('## Graph & Adduct dataset completed')
    
    ECC_Model = load_Model_from_file(mfileh5)
    # print('## Model loading completed')
    
    re = predict(ECC_Model,param.adduct_SET,DataSet,adduct,)
    data = {'SMILES' : smiles,
            'Adduct' : adduct,
            'Ture CCS': ccs,
            'Predicted CCS':re}
    df = DataFrame(data)
    df.to_csv(ofile,index=False)
    # print('## CCS predicted completed')
    if Isevaluate == 1:
        re_Bais = Bais(ccs,re)
    return re_Bais

def Model_train(ifile, ParameterPath, ofile, EPOCHS, BATCHS, Vis, All_Atoms=[], adduct_SET=[]):
    '''
    * Train
    *
    * Attributes
    * ----------
    * ifile         : File path for storing the data of smiles and adduct
    * ParameterPath : Save path of related data parameters
    * ofile         : File path where the model is stored
    '''
    # Read the smiles adduct CCS in the file
    smiles, adduct, ccs = read_data(ifile)
    # print('## Read data : ',len(smiles))
    
    # If the user does not enter the number of elements, then the default is the set of all elements in the training set
    if len(All_Atoms) == 0:
        All_Atoms = GetSmilesAtomSet(smiles) # Calculate the set of elements used in the training set
    
    # 3D conformation of the input SMILES
    smiles, adduct, ccs, Coordinate = Generating_coordinates(smiles, adduct, ccs, All_Atoms)
    
    # Data normalization of the generated coordinate data
    ALL_DATA = []
    for i in Coordinate:
        for ii in i:
            ALL_DATA.append(ii[0]);ALL_DATA.append(ii[1]);ALL_DATA.append(ii[2]);   
    Max_Coor, Min_Coor = np.max(ALL_DATA), np.min(ALL_DATA)
    
    for i in range(len(Coordinate)):
        Coordinate[i] = (np.array(Coordinate[i]) - Min_Coor) / (Max_Coor - Min_Coor)
    # Adduct set
    if len(adduct_SET) == 0:
        adduct_SET = list(set(list(adduct)))
        adduct_SET.sort()
    
    # print('## All element types : ', All_Atoms)
    # print('## All adduct types : ', adduct_SET)
    
    # Storing parameters in objects
    rw = Parameter(adduct_SET, All_Atoms, Max_Coor, Min_Coor)
    output_hal = open(ParameterPath, 'wb')
    output_hal.write(pickle.dumps(rw))
    output_hal.close()
    
    # Construct Graph from the input data
    adj, features, edge_features = convertToGraph(smiles, Coordinate, All_Atoms)
    DataSet = MyDataset(features, adj, edge_features, ccs)
    #print('## Build graph data successfully')
    #print('## Dataset:',DataSet,' Node features:',DataSet[0].x.shape,' Edge features:',DataSet[0].e.shape,)
    # Production of models for training
    ECC_Model = Mymodel(DataSet,adduct_SET)
    # Training Model
    ECC_Model = train(ECC_Model,DataSet,adduct,adduct_SET, EPOCHS = EPOCHS, BATCHS = BATCHS, Vis = Vis)
    # Save model
    ECC_Model.save(ofile)
    return ECC_Model
