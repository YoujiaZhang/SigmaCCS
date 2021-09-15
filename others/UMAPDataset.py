from tqdm import *
import time, random
import sys

from rdkit.Chem import AllChem
from rdkit import DataStructs
import rdkit.Chem as Chem

from pandas import Series,DataFrame
import pandas as pd
import numpy as np

elements = set(['As', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', 'Se'])
All_Atoms = ['As', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', 'Se']

Atom_radius = {'N': 0.38613861386138615, 'Se': 0.8316831683168316, 'F': 0.31683168316831684, 'Co': 0.7821782178217822, 'O': 0.3069306930693069, 'As': 0.8811881188118812, 'Br': 0.8118811881188119, 'Cl': 0.6633663366336634, 'S': 0.7029702970297029, 'C': 0.42574257425742573, 'P': 0.7821782178217822, 'I': 1.0, 'H': 0.0}
Atom_mass = {'N': 0.1032498671726695, 'Se': 0.6191756039662093, 'F': 0.14289880110277858, 'Co': 0.46010207747584464, 'As': 0.5870984688775774, 'O': 0.11907762668280056, 'Br': 0.6266738249259133, 'Cl': 0.2735981682735815, 'S': 0.24668718033769474, 'C': 0.08739526021884797, 'P': 0.2380194434270746, 'I': 1.0, 'H': 0.0}
    
Max_Coor =  15.615155868453662
Min_Coor = -15.475082312818216

def atom_feature(atom,Coordinate,All_Atoms,Atom_radius,Atom_mass):
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol() ,All_Atoms) +
        one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4]) +
        [Atom_radius[atom.GetSymbol()],Atom_mass[atom.GetSymbol()]] +
        one_of_k_encoding_unk(atom.IsInRing(), [0, 1]) +
        Coordinate
    )

def edge_feature(iMol,iAdjTmp):
    Edge_feature = []
    for bond in iMol.GetBonds():
        bond_feature = np.array(
            one_of_k_encoding_unk(bond.GetBondTypeAsDouble(),[1,1.5,2,3])
        )
        Edge_feature.append(bond_feature)
        Edge_feature.append(bond_feature)
    Edge_feature = np.array(Edge_feature)
    Edge_feature = Edge_feature.astype(np.float)
    return Edge_feature

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def Constructed_graph_dataset(data):
    adj, features, edge_features = [], [], []
    NodeNumFeatures, EdgeNumFeatures = 0, 4
    DF_index = []

    p_bar = tqdm(range(len(list(data['SMILES']))), desc="Constructs", total=len(list(data['SMILES'])), ncols=90,)
    for i, p_bar_i in zip(data.iterrows(), p_bar):
        index, row = i[0], i[1]
        try:
            iMol3D = Chem.MolFromMolFile('data/UMAP data/Coordinate data/'+str(row['Pubchem ID'])+'.mol')
            maxNumAtoms = iMol3D.GetNumAtoms()
            iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol3D)
        except Exception as e:
            print(e)
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

    p_bar.close()
    
    return features, adj, edge_features