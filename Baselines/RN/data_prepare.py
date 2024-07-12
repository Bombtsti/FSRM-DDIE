import os
import pickle
import sqlite3
import pandas as pd
import torch
from rdkit import Chem
import numpy as np
from torch_geometric.data import Data

SMILE_SHAPE = 3535


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom,
                explicit_H=True,
                use_chirality=False):

    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C','N','O', 'S','F','Si','P', 'Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl',
            'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H', 'Li','Ge','Cu','Au','Ni','Cd','In',
            'Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
        ]) + [atom.GetDegree()/10, atom.GetImplicitValence(),
                atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
                ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if explicit_H:
        results = results + [atom.GetTotalNumHs()]

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)


def get_mol_edge_list_and_feat_mtx(mol_graph):
    features = [(atom.GetIdx(), atom_features(atom)) for atom in mol_graph.GetAtoms()]
    features.sort()  # to make sure that the feature matrix is aligned according to the idx of the atom
    _, features = zip(*features)
    features = torch.stack(features)

    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol_graph.GetBonds()])
    undirected_edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list

    # assert TOTAL_ATOM_FEATS == features.shape[-1], "Expected atom n_features and retrived n_features not matching"
    return undirected_edge_list.T, features


def generate_graph_data(mol_graph):
    MOL_EDGE_LIST_FEAT_MTX = get_mol_edge_list_and_feat_mtx(mol_graph)
    # TOTAL_ATOM_FEATS = (next(iter(MOL_EDGE_LIST_FEAT_MTX.values()))[1].shape[-1])
    edge_index = MOL_EDGE_LIST_FEAT_MTX[0]
    features = MOL_EDGE_LIST_FEAT_MTX[1]
    data = Data(x=features, edge_index=edge_index)
    return data

def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nData saved as {filename}!')


def test():
    # conn=sqlite3.connect("../METADDIEdata/Drug_META_DDIE.db")
    # smile = {}
    # smileFile = pd.read_sql("select * from drug",conn)
    # for i in range(SMILE_SHAPE):
    #     smile[smileFile.loc[i][0]] = (smileFile.loc[i][3])
    # print(len(smile))

    with open('./drug_graph.pkl', 'rb') as f:
        a = pickle.load(f)
        # print(a)
        print(a['DB01100'])
        print(a['DB00976'])

if __name__ == '__main__':
    # smiles = {}
    # mols = {}
    # conn=sqlite3.connect("../METADDIEdata/Drug_META_DDIE.db")
    # smileFile = pd.read_sql("select * from drug",conn)
    # for i in range(SMILE_SHAPE):
    #     smiles[smileFile.loc[i][0]] = (smileFile.loc[i][3])
    #     # drug_id_mol_graph_tup = [(smileFile.loc[i][0], Chem.MolFromSmiles(smileFile.loc[i][3].strip()))]
    #     mols[smileFile.loc[i][0]] = generate_graph_data(Chem.MolFromSmiles(smileFile.loc[i][3].strip()))
    # save_data(mols,"./drug_graph.pkl")
    test()







