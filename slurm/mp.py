import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from rdkit.Chem import AllChem
from rdkit import Chem

from multiprocessing import Pool
from multiprocessing import cpu_count

from spektral.data import Dataset,Graph
from spektral.layers import GlobalSumPool,ECCConv
from spektral.data import BatchLoader

from pandas import Series,DataFrame

import pandas as pd
import numpy as np
import time

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
print('tesorflow version :',tf.__version__)

# Load the completed training model
ECC_Model = load_model('model/model.h5',
                       custom_objects = {"ECCConv": ECCConv,
                                         "GlobalSumPool": GlobalSumPool})
print('model load succ')

# Set the relevant parameters for ETKDG conformation
ps = AllChem.ETKDGv3()
ps.randomSeed = -1
ps.maxAttempts = 1
ps.useRandomCoords = True
ps.numThreads = 1
elements = set(['As', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', 'Se'])
All_Atoms = ['As', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', 'Se']

# Relevant parameters of the model
Atom_radius = {'N': 0.38613861386138615, 'Se': 0.8316831683168316, 'F': 0.31683168316831684, 'Co': 0.7821782178217822, 'O': 0.3069306930693069, 'As': 0.8811881188118812, 'Br': 0.8118811881188119, 'Cl': 0.6633663366336634, 'S': 0.7029702970297029, 'C': 0.42574257425742573, 'P': 0.7821782178217822, 'I': 1.0, 'H': 0.0}
Atom_mass = {'N': 0.1032498671726695, 'Se': 0.6191756039662093, 'F': 0.14289880110277858, 'Co': 0.46010207747584464, 'As': 0.5870984688775774, 'O': 0.11907762668280056, 'Br': 0.6266738249259133, 'Cl': 0.2735981682735815, 'S': 0.24668718033769474, 'C': 0.08739526021884797, 'P': 0.2380194434270746, 'I': 1.0, 'H': 0.0}

Max_Coor =  15.615155868453662 
Min_Coor = -15.475082312818216

def atom_feature(atom,Coordinate,All_Atoms,Atom_radius,Atom_mass):
    '''The functions are defined in the sigma/GraphData.py file'''
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol() ,All_Atoms) +
        one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4]) +
        [Atom_radius[atom.GetSymbol()],Atom_mass[atom.GetSymbol()]] +
        one_of_k_encoding_unk(atom.IsInRing(), [0, 1]) +
        Coordinate
    )

def edge_feature(iMol,iAdjTmp):
    '''The functions are defined in the sigma/GraphData.py file'''
    Edge_feature = []
    count = 0
    for bond in iMol.GetBonds():
        count += 1
        bond_feature = np.array(
            one_of_k_encoding_unk(bond.GetBondTypeAsDouble(),[1,1.5,2,3])
        )
        Edge_feature.append(bond_feature)
        Edge_feature.append(bond_feature)
    Edge_feature = np.array(Edge_feature)
    Edge_feature = Edge_feature.astype(np.float)
    return Edge_feature

def one_of_k_encoding_unk(x, allowable_set):
    '''The functions are defined in the sigma/GraphData.py file'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

class MyDataset(Dataset):
    '''The functions are defined in the sigma/GraphData.py file'''
    def __init__(self, features, adj, edge_features, ccs, **kwargs):
        self.features = features
        self.adj = adj
        self.edge_features = edge_features
        self.ccs = ccs
        super().__init__(**kwargs)
        
    def read(self):
        return [Graph(x = self.features[i],
                      a = self.adj[i],
                      e = self.edge_features[i],
                      y = float(self.ccs[i])) for i in range(len(self.adj))]

def construct_3d(smiles):
    '''
    * Storage path for mol files containing 3D information
    *
    * Attributes
    * ----------
    * smiles      : The SMILES string list of the molecule
    '''
    PATH = 'BigData/MOL/'+str(smiles[1])+'.mol'  
    if os.path.exists(PATH):
        return
    try:
        iMol3D = Chem.MolFromSmiles(smiles[0])
        atoms = [atom.GetSymbol() for atom in iMol3D.GetAtoms()]
        bonds = [bond for bond in iMol3D.GetBonds()]
        if len(atoms) == 1 and len(bonds) <= 1:
            return
        iMol3D = Chem.AddHs(iMol3D)
        re = AllChem.EmbedMultipleConfs(iMol3D, numConfs = 1, params = ps)
        if len(re) != 0:
            re = AllChem.MMFFOptimizeMoleculeConfs(iMol3D,  numThreads = 0)
        print(Chem.MolToMolBlock(iMol3D),file=open('BigData/MOL/'+str(smiles[1])+'.mol','w+'))
    except:
        return    

def predict_adduct(Model,adduct_SET,dataset,adduct):
    '''The functions are defined in the sigma/model.py file'''
    loader = BatchLoader(dataset,batch_size=1,epochs=1,shuffle=False)
    loader_data = ()
    ltd_index = 0
    for i in loader.load():
        adduct_one_hot = [one_of_k_encoding_unk(adduct[ltd_index+ltd_index_i],adduct_SET) for ltd_index_i in range(len(i[1]))]
        adduct_one_hot = np.array(adduct_one_hot)
        one_sample = ((adduct_one_hot,i[0][0],i[0][1],i[0][2]),i[1])
        loader_data += (one_sample,)
        ltd_index += 1
    loader_data = (i for i in loader_data)
    
    y_pred = []
    for batch in loader_data:
        inputs, target = batch
        predictions = Model(inputs, training=False) # predict
        pred = np.array(predictions[0])
        y_pred.append(pred[0])
    return y_pred


def PRE(ID,ccs = None,adduct=None):
    '''
    * Multi-core parallel construction of graph datasets of molecules and then prediction of ccs for each molecule
    *
    * Attributes
    * ----------
    * ID          :The Pubchem CID for each molecule
    '''
    DF_index = []
    adj, features, edge_features = [], [], []
    NodeNumFeatures, EdgeNumFeatures = 0, 4
    for id_ in ID:
        try:
            iMol3D = Chem.MolFromMolFile('BigData/MOL/'+str(id_)+'.mol')
            maxNumAtoms = iMol3D.GetNumAtoms()
            iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol3D)
        except:
            DF_index.append(False)
            continue;
        DF_index.append(True)
        one_edge_features = edge_feature(iMol3D,iAdjTmp)
        edge_features.append(one_edge_features)
        
        iFeature = np.zeros((maxNumAtoms, NodeNumFeatures))
        iFeatureTmp = []
        for atom in iMol3D.GetAtoms():
            Coord = list(iMol3D.GetConformer().GetAtomPosition(atom.GetIdx()))
            Coord = list((np.array(Coord) - Min_Coor)/(Max_Coor - Min_Coor))
            iFeatureTmp.append(atom_feature(atom,Coord,All_Atoms,Atom_radius,Atom_mass))
        features.append(np.array(iFeatureTmp))
        adj.append(iAdjTmp)
        
    features = np.asarray(features)
    edge_features = np.asarray(edge_features)
    
    if ccs == None:
        ccs = [0 for i in range(len(adj))]
    DataSet = MyDataset(features, adj, edge_features, ccs)  
    print(DataSet) # Output the number of constructed graphs
    
    adduct_1 = ['[M+H]+'  for i in range(len(ID))]
    adduct_2 = ['[M+Na]+' for i in range(len(ID))]
    adduct_3 = ['[M-H]-'  for i in range(len(ID))]

    re_1 = predict_adduct(ECC_Model,['[M+H]+', '[M+Na]+', '[M-H]-'],DataSet,adduct_1)
    re_2 = predict_adduct(ECC_Model,['[M+H]+', '[M+Na]+', '[M-H]-'],DataSet,adduct_2)
    re_3 = predict_adduct(ECC_Model,['[M+H]+', '[M+Na]+', '[M-H]-'],DataSet,adduct_3)
    
    return DF_index, re_1, re_2, re_3


if __name__ == '__main__':
    # Serial number of the output file
    INDEX = 1 
    
    # Constructing pubchem data file names
    file_names = []
    for t in range(1,1500000,500000):
        number = str(t).zfill(9) + '_' + str(t+500000-1).zfill(9)
        file_names.append(number)
    
    # Read each file and merge it into one large file
    All_data = []
    for filename in file_names:
        data = pd.read_csv('data/Compound_'+filename+'.csv')
        All_data.append(data)
    data = pd.concat(All_data, axis=0, ignore_index=True)
    # print(len(data))
    
    data = data[['ISO SMILES','InChi','Inchikey','Molecular weight','Pubchem ID']]
    smiles = list(data['ISO SMILES'])
    ID     = list(data['Pubchem ID'])
    
    t = time.time()
    # Parallel Computing. https://docs.python.org/dev/library/multiprocessing.html
    with Pool(48) as p:
        p.map(construct_3d, [(smiles[i],ID[i]) for i in range(len(smiles))])
        
    DF_index, re_1, re_2, re_3 = PRE(ID,)
    data = data[Series(DF_index)]
    data['[M+H]+']  = re_1
    data['[M+Na]+'] = re_2
    data['[M-H]-']  = re_3
    
    data.to_csv('BigData/pred/'+str(INDEX)+'.csv',index=False)
    
    e = time.time() - t
    print(f"one: {e/3/len(smiles)}")
    print(f"All: {e}",len(smiles))