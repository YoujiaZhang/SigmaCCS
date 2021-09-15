import pandas as pd
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from spektral.data import Dataset,Graph
import pickle
import sigma.parameter as parameter

from tqdm import *
import time, random
import sys

def read_data(filename,):
    '''
    * Attributes
    * ----------
    * filename : input file
    *
    * Returns
    * -------
    * smiles : SMILES
    * adduct : Adduct ([M+H]+,[M+Na]+,[M-H]]-)
    * ccs    : CCS
    '''
    data = pd.read_csv(filename)
    smiles = np.array(data['SMILES'])
    adduct = np.array(data['Adduct'])
    ccs    = np.array(data['True CCS'])
    return smiles, adduct, ccs

def Standardization(data):
    '''
    * Attributes
    * ----------
    * data : Data that need to be normalized
    *
    * Returns
    * -------
    * data : Normalized data
    '''
    data_list = [data[i] for i in data]
    Max_data, Min_data = np.max(data_list), np.min(data_list)
    for i in data:
        data[i] = (data[i] - Min_data) / (Max_data - Min_data)
    return data

def Generating_coordinates(smiles, adduct, ccs, All_Atoms, ps = AllChem.ETKDGv3(),):
    '''
    * Using ETKDG to generate 3D coordinates of molecules
    *
    * Attributes
    * ----------
    * smiles    : The SMILES string of the molecule
    * adduct    : Adduct of molecules
    * ccs       : CCS of molecules
    * All_Atoms : Element set (The type of element provided must cover all elements contained in the molecule)
    * ps        : ETKDG algorithm provided by RDkit
    *
    * Returns
    * -------
    * succ_smiles : SMILES of The molecules with 3D conformation can be successfully generated
    * succ_adduct : Adduct of The molecules with 3D conformation can be successfully generated
    * succ_ccs    : CCS of The molecules with 3D conformation can be successfully generated
    * Coordinate  : 3D coordinates of molecules
    '''
    succ_smiles = []
    succ_adduct = []
    succ_ccs    = []
    Coordinate  = []
    
    p_bar = tqdm(range(len(smiles)), desc="The generation of 3d conformers", total=len(smiles), ncols=90, file=sys.stdout)
    INDEX = -1    
    for smi, p_bar_i in zip(smiles, p_bar):
        INDEX += 1
        try:
            iMol = Chem.MolFromSmiles(smi)
            iMol = Chem.RemoveHs(iMol)
        except:
            continue;
            
        atoms = [atom.GetSymbol() for atom in iMol.GetAtoms()]
        bonds = [bond for bond in iMol.GetBonds()]
        # Is the number of atoms greater than 1
        if len(atoms) == 1 and len(bonds) <= 1:
            continue;
        # Determine whether the element is in all_ In atoms
        Elements_not_included = 0
        for atom in atoms:
            if atom not in All_Atoms:
                Elements_not_included = 1
        if Elements_not_included == 1:
            continue;
        # Adding H to a molecular object
        iMol3D = Chem.AddHs(iMol)
        
        # The 3D conformation of the generating molecule
        ps.randomSeed = -1
        ps.maxAttempts = 1
        ps.numThreads = 0
        ps.useRandomCoords = True
        re = AllChem.EmbedMultipleConfs(iMol3D, numConfs = 1, params = ps)
        # Whether the conformation is successful or not
        if len(re) == 0:
            print('conformation is error')
            continue;
        # MMFF94
        re = AllChem.MMFFOptimizeMoleculeConfs(iMol3D,  numThreads = 0)

        This_mol_Coordinate = []
        for atom in iMol3D.GetAtoms():
            Coord = list(iMol3D.GetConformer().GetAtomPosition(atom.GetIdx()))
            This_mol_Coordinate.append(Coord)
        Coordinate.append(This_mol_Coordinate)
        
        succ_smiles.append(smi)
        succ_adduct.append(adduct[INDEX])
        succ_ccs.append(ccs[INDEX])
        
    p_bar.close()
    return succ_smiles, succ_adduct, succ_ccs,  Coordinate

def GetSmilesAtomSet(smiles):
    '''
    * Gets the collection of all elements in the dataset
    *
    * Attributes
    * ----------
    * smiles    : The SMILES string of the molecule
    *
    * Returns
    * -------
    * All_Atoms : Element set
    '''
    All_Atoms = []
    for i in range(len(smiles)):
        mol = Chem.MolFromSmiles(smiles[i])
        All_Atoms += [atom.GetSymbol() for atom in mol.GetAtoms()]
        All_Atoms = list(set(All_Atoms))
    All_Atoms.sort()
    return All_Atoms

def convertToGraph(smi_lst,Coordinate,All_Atoms):
    '''
    * Construct a graph dataset for the input molecular dataset
    *
    * Attributes
    * ----------
    * smi_lst    : The SMILES string list of the molecule
    * Coordinate : The coordinate data of each molecule
    * All_Atoms  : A Set of all elements in a SMILES dataset
    *
    * Returns
    * -------
    * adj           : Adjacency matrix
    * features      : The feature vector of each node(Atom) in the graph
    * edge_features : The feature vector(one-hot encode) of each edge(Bond) in the graph
    '''
    ##################################################
    # !!!Note!!!: you need to record the radius and mass of the atoms that exist in the atomic set
    ##################################################
    # The atomic radius and atomic mass are normalized
    Atom_radius = Standardization(parameter.Atom_radius)
    Atom_mass   = Standardization(parameter.Atom_mass)
    #print(Atom_radius)
    #print(Atom_mass)
    
    adj, features, edge_features = [], [], []
    NodeNumFeatures, EdgeNumFeatures, INDEX = 0, 4, -1
    # Traverses all the SMILES strings
    for smi in smi_lst:
        INDEX += 1
        iMol = Chem.MolFromSmiles(smi)    # Converts a SMILES string to a MOL object
        maxNumAtoms = iMol.GetNumAtoms()  # by zhangyoujia
        iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol) # The adjacency matrix of MOL is obtained
        # Characteristics of structural chemical bonds(Edge)
        one_edge_features = edge_feature(iMol)
        edge_features.append(one_edge_features)
        # Constructing Atoms(Nodes) feature data
        iFeature = np.zeros((maxNumAtoms, NodeNumFeatures))
        iFeatureTmp = []
        # Construct vectors for each atom of the molecule
        for atom in iMol.GetAtoms():
            iFeatureTmp.append(atom_feature(atom,INDEX,Coordinate,All_Atoms,Atom_radius,Atom_mass))
        features.append(np.array(iFeatureTmp))
        adj.append(iAdjTmp)
        
    features = np.asarray(features)
    edge_features = np.asarray(edge_features)
    return adj, features, edge_features

def atom_feature(atom,INDEX,Coordinate,All_Atoms,Atom_radius,Atom_mass):
    '''
    * Component atom vector
    *
    * Attributes
    * ----------
    * atom        : Atom object
    * INDEX       : The molecule to which the atom belongs and the index of the molecule in SMILES list
    * Coordinate  : The 3D coordinates of all atoms of each molecule
    * All_Atoms   : A Set of all elements in a SMILES dataset
    * Atom_radius : Atomic radius dictionary
    * Atom_mass   : Atomic mass dictionary
    *
    * Returns
    * -------
    * adj           : Adjacency matrix
    * features      : The feature vector of each node(Atom) in the graph
    * edge_features : The feature vector(one-hot encode) of each edge(Bond) in the graph
    '''
    return np.array(
        # Atomic Type (One-Hot)
        one_of_k_encoding_unk(atom.GetSymbol() ,All_Atoms) +
        # Atomic Degree (One-Hot)
        one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4]) +
        # Atomic radius  Atomic mass (float)
        [Atom_radius[atom.GetSymbol()],Atom_mass[atom.GetSymbol()]] +
        # Atomic is in Ring ? (One-Hot)
        one_of_k_encoding_unk(atom.IsInRing(), [0, 1]) +
        # Coordinate (float)
        list(Coordinate[INDEX][atom.GetIdx()])
    )

def one_of_k_encoding_unk(x, allowable_set):
    '''
    * One-hot encoding
    *
    * Attributes
    * ----------
    * x             : Data
    * allowable_set : Data range
    '''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def edge_feature(iMol):
    '''
    * Constructing edge feature matrix
    *
    * Attributes
    * ----------
    * iMol : Molecular objects
    *
    * Returns
    * -------
    * Edge_feature : Edge feature matrix of molecules
    '''
    iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol) # The adjacency matrix of MOL is obtained
    Edge_feature = []
    count = 0
    for bond in iMol.GetBonds():
        count += 1
        bond_feature = np.array(
            one_of_k_encoding_unk(bond.GetBondTypeAsDouble(),[1,1.5,2,3]) 
            # One-hot 1.0 for SINGLE, 1.5 for AROMATIC, 2.0 for DOUBLE
        )
        Edge_feature.append(bond_feature)
        Edge_feature.append(bond_feature)
    Edge_feature = np.array(Edge_feature)
    Edge_feature = Edge_feature.astype(np.float)
    return Edge_feature
        
class MyDataset(Dataset):
    '''
    * Constructing edge feature matrix
    *
    * Attributes
    * ----------
    * features      : Node feature matrix
    * adj           : Adjacency matrix
    * edge_features : Edge feature matrix
    * ccs           : CCS of molecules
    '''
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