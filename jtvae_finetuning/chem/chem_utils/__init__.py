""" Contains many chem utils codes """
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import rdkit
from rdkit import Chem
from rdkit.Chem import Crippen
import networkx as nx
from rdkit.Chem import rdmolops
from rdkit.Chem import QED
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds
from rdkit.Chem import AllChem
import os
import pickle
import numpy as np
from sklearn import svm
import torch
from tqdm import tqdm
from tqdm.auto import trange
import pandas as pd
from mordred import Calculator, descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors

# My imports
from jtvae_finetuning.chem.chem_utils.SA_Score import sascorer
from jtvae_finetuning.chem.chem_utils.NP_Score import npscorer
from jtvae_finetuning.chem.jtnn.jtnn_vae import JTNNVAE
from jtvae_finetuning.chem.chem_model import JTVAE
from jtvae_finetuning.chem.jtnn import Vocab
from jtvae_finetuning.chem.jtnn.datautils import tensorize
print(os.path.dirname(os.path.realpath(__file__)))
MODULE_DIR = os.path.dirname(os.path.realpath(__file__))

with open(os.path.dirname(os.path.realpath(__file__))+'/GB_DN.pkl', "rb") as f:
   DN_model = pickle.load(f)
with open(os.path.dirname(os.path.realpath(__file__))+'/RR_DM.pkl', "rb") as f:
   DM_model = pickle.load(f)
# with open(os.path.dirname(os.path.realpath(__file__))+'/RR_AN.pkl', "rb") as f:
#    AN_model = pickle.load(f)

# Make rdkit be quiet
def rdkit_quiet():
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

def get_mol(smiles_or_mol):                                                     
    '''                                                                                                                                       
    Loads SMILES/molecule into RDKit's object                                   
    '''                                                                                                                                       
    if isinstance(smiles_or_mol, str):                                          
        if len(smiles_or_mol) == 0:                                              
            return None                                                           
        mol = Chem.MolFromSmiles(smiles_or_mol)                                 
        if mol is None:                                                          
            return None                                                           
        try:                                                                    
            Chem.SanitizeMol(mol)                                                 
        except ValueError:                                                      
            return None                                                           
        return mol                                                              
    return smiles_or_mol

def standardize_smiles(smiles):
    """ Get standard smiles without stereo information """
    mol = get_mol(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def penalized_logP(smiles: str, min_score=-float("inf")) -> float:
    """ calculate penalized logP for a given smiles string """
    mol = Chem.MolFromSmiles(smiles)
    logp = Crippen.MolLogP(mol)
    sa = SA(mol)

    # Calculate cycle score
    cycle_length = _cycle_score(mol)

    """
    Calculate final adjusted score.
    These magic numbers are the empirical means and
    std devs of the dataset.

    I agree this is a weird way to calculate a score...
    but this is what previous papers did!
    """
    score = (
        (logp - 2.45777691) / 1.43341767
        + (-sa + 3.05352042) / 0.83460587
        + (-cycle_length - -0.04861121) / 0.28746695
    )
    return max(score, min_score)

def logP_func(smiles: str) -> float:
    """ calculate logP for a given smiles string """
    mol = Chem.MolFromSmiles(smiles)
    logp = Crippen.MolLogP(mol)
    
    return logp

def SAS_func(smiles: str) -> float:
    """ calculate SAS for a given smiles string """
    mol = Chem.MolFromSmiles(smiles)
    sa = SA(mol)

    return sa

def tpsa_func(smiles: str) -> float:
    """ calculate TPSA for a given smiles string """
    mol = Chem.MolFromSmiles(smiles)
    tpsa = CalcTPSA(mol)

    return tpsa

def num_rb_func(smiles: str) -> float:
    """ calculate number of rotatable bonds for a given smiles string """
    mol = Chem.MolFromSmiles(smiles)
    num_rb = CalcNumRotatableBonds(mol)

    return num_rb

# def qed_func(smiles: str) -> float:
#     """ calculate SAS for a given smiles string """
#     mol = Chem.MolFromSmiles(smiles)
#     qed = QED(mol)

#     return qed

def qed_func(smiles: str) -> float:
    """ calculate qed for a given smiles string """
    mol = Chem.MolFromSmiles(smiles)
    
    return QED.qed(mol)


def NP_score_func(smiles: str) -> float:
    """ calculate NP_score for a given smiles string """
    mol = Chem.MolFromSmiles(smiles)
    
    NP_score = npscorer.scoreMol(mol)
    
    return NP_score


def fingerprints_from_mol( mol):
    fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
    size = 2048
    nfp = np.zeros((1, size), np.int32)
    for idx,v in fp.GetNonzeroElements().items():
        nidx = idx%size
        nfp[0, nidx] += int(v)
    return nfp


def SA(mol):
    return sascorer.calculateScore(mol)


def _cycle_score(mol):
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return cycle_length

# def QED(smiles: str) -> float:S
#     mol = Chem.MolFromSmiles(smiles)
#     return Chem.QED.qed(mol)
activity_clf = None
activity_model = None


def _load_activity_assets():
    """Load legacy DRD2 assets only when activity_func is called."""
    global activity_clf, activity_model
    if activity_clf is None:
        with open(os.path.join(MODULE_DIR, "DRD2.pkl"), "rb") as f:
            activity_clf = pickle.load(f)
    if activity_model is None:
        with open("runtime_assets/data/chem/zinc/orig_model/vocab.txt") as f:
            vocab = Vocab([x.strip() for x in f.readlines()])
        activity_model = JTVAE.load_from_checkpoint(
            "runtime_assets/pretrained_models/chem.ckpt", vocab=vocab
        )


"""
def activity_func(smiles: str) -> float:
    _load_activity_assets()
    
    descriptors = ['FractionCSP3', 'BCUT2D_MRLOW', 'fr_quatN', 'PEOE_VSA8', 'fr_Ar_N',
       'MinEStateIndex', 'fr_NH0', 'fr_imidazole', 'fr_bicyclic',
       'fr_para_hydroxylation', 'BCUT2D_MWHI', 'PEOE_VSA2',
       'NumSaturatedHeterocycles', 'BalabanJ', 'SlogP_VSA11',
       'NumAliphaticHeterocycles', 'PEOE_VSA5', 'MaxAbsEStateIndex',
       'NumAromaticHeterocycles', 'MaxAbsPartialCharge', 'VSA_EState3',
       'VSA_EState6', 'VSA_EState2', 'HallKierAlpha', 'PEOE_VSA9',
       'fr_aniline', 'fr_pyridine', 'VSA_EState4', 'PEOE_VSA1', 'Chi4v',
       'fr_C_O', 'EState_VSA4', 'SlogP_VSA3', 'MolLogP', 'fr_Nhpyrrole',
       'Chi1v', 'fr_NH1', 'EState_VSA3', 'PEOE_VSA7', 'fr_ether', 'SMR_VSA10',
       'SlogP_VSA8', 'SMR_VSA1', 'HeavyAtomMolWt']
    if not isinstance(smiles, str):
        raise ValueError("SMILES input should be a string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    all_vec = model.jtnn_vae.encode_from_smiles([smiles])
    all_vec = torch.cat(all_vec, dim=1)
    tree_vec, mol_vec = np.hsplit(all_vec, 2)
    tree_, _ = model.jtnn_vae.rsample(tree_vec, model.jtnn_vae.T_mean, model.jtnn_vae.T_var)
    mol_, _ = model.jtnn_vae.rsample(mol_vec, model.jtnn_vae.G_mean, model.jtnn_vae.G_var)

    tree_vec = tree_.data.cpu().numpy().reshape(1, -1)
    mol_vec = mol_.data.cpu().numpy().reshape(1, -1)
    z_vec = np.hstack((tree_vec, mol_vec))
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors)
    desc = np.array(desc_calc.CalcDescriptors(mol)).reshape(1, -1) 
    imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    desc = imputer.fit_transform(desc)
    zd = np.hstack((z_vec, desc)) 
    X_scaler = MinMaxScaler()
    zd = X_scaler.fit_transform(zd)
    prediction = activity_clf.predict(zd)
    return float(prediction)
    """
      
def activity_func(smiles: str) -> float:
    _load_activity_assets()
    descriptors = ['FractionCSP3', 'BCUT2D_MRLOW', 'fr_quatN', 'PEOE_VSA8', 'fr_Ar_N',
                   'MinEStateIndex', 'fr_NH0', 'fr_imidazole', 'fr_bicyclic',
                   'fr_para_hydroxylation', 'BCUT2D_MWHI', 'PEOE_VSA2',
                   'NumSaturatedHeterocycles', 'BalabanJ', 'SlogP_VSA11',
                   'NumAliphaticHeterocycles', 'PEOE_VSA5', 'MaxAbsEStateIndex',
                   'NumAromaticHeterocycles', 'MaxAbsPartialCharge', 'VSA_EState3',
                   'VSA_EState6', 'VSA_EState2', 'HallKierAlpha', 'PEOE_VSA9',
                   'fr_aniline', 'fr_pyridine', 'VSA_EState4', 'PEOE_VSA1', 'Chi4v',
                   'fr_C_O', 'EState_VSA4', 'SlogP_VSA3', 'MolLogP', 'fr_Nhpyrrole',
                   'Chi1v', 'fr_NH1', 'EState_VSA3', 'PEOE_VSA7', 'fr_ether', 'SMR_VSA10',
                   'SlogP_VSA8', 'SMR_VSA1', 'HeavyAtomMolWt']
    
    if not isinstance(smiles, str):
        raise ValueError("SMILES input should be a string")
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    try:
        # Try to encode the SMILES string to latent space vectors
        all_vec = activity_model.jtnn_vae.encode_from_smiles([smiles])
    except Exception as e:
        print(f"Error in encoding SMILES {smiles}: {e}")
        return 0.0  # If there's an error, return 0 as the prediction
    
    all_vec = torch.cat(all_vec, dim=1)
    tree_vec, mol_vec = np.hsplit(all_vec, 2)
    tree_, _ = activity_model.jtnn_vae.rsample(
        tree_vec, activity_model.jtnn_vae.T_mean, activity_model.jtnn_vae.T_var
    )
    mol_, _ = activity_model.jtnn_vae.rsample(
        mol_vec, activity_model.jtnn_vae.G_mean, activity_model.jtnn_vae.G_var
    )

    tree_vec = tree_.data.cpu().numpy().reshape(1, -1)
    mol_vec = mol_.data.cpu().numpy().reshape(1, -1)
    z_vec = np.hstack((tree_vec, mol_vec))
    
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors)
    desc = np.array(desc_calc.CalcDescriptors(mol)).reshape(1, -1)
    
    imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    desc = imputer.fit_transform(desc)
    
    zd = np.hstack((z_vec, desc))
    z_all = pd.read_csv("zpred_vec.csv", header=None).values
    desc_all = pd.read_csv("desc_pred.csv", header=None).values
    zd_all = np.hstack((z_all, desc_all))
    
    X_scaler = MinMaxScaler()
    X_scaler.fit(zd_all)
    zd = X_scaler.transform(zd)
    
    prediction = activity_clf.predict(zd)
    return float(prediction)

def DN_func(smiles: str) -> float:
    FEATURES_USED = ['nN', 'ATS0dv', 'AATS5dv', 'AATS5s', 'AATS0se',
        'ATSC0c', 'ATSC1dv', 'ATSC1s', 'ATSC1v', 'AATSC1i',
        'GATS2d', 'GATS2s', 'GATS1v', 'GATS1se', 'BCUTv-1l',
        'RNCG', 'RPCG', 'AXp-5d', 'SIC1', 'PEOE_VSA10']
    
    # 选择需要的描述符
    calc_all = Calculator(descriptors, ignore_3D=True)
    selected_descriptors = [d for d in calc_all.descriptors if str(d) in FEATURES_USED]
    calc = Calculator(selected_descriptors, ignore_3D=True)

    # 构建标准化器
    df_x = pd.read_csv(os.path.join(MODULE_DIR, "df_final_DN.csv"))
    X_all = df_x[FEATURES_USED].values
    scaler = StandardScaler()
    X_scaled_all = scaler.fit(X_all)

    # 分子解析
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("无效的 SMILES：无法解析为分子对象")
    
    try:
        desc = calc(mol)
        desc_values = np.array([desc[str(d)] for d in calc.descriptors], dtype=float).reshape(1, -1)
    except Exception as e:
        raise ValueError(f"描述符计算失败：{e}")
    
    # 处理 NaN：将所有 NaN 替换为 0
    if np.isnan(desc_values).any():
        desc_values = np.nan_to_num(desc_values, nan=0.0)

    # 标准化 + 预测
    desc_scaled = scaler.transform(desc_values)
    prediction = DN_model.predict(desc_scaled)
    return float(prediction)

def DM_func(smiles: str) -> float:
    FEATURES_USED = ['ABC', 'ATS0dv', 'AATS2d', 'AATS6s', 'AATS0se', 'ATSC1dv', 'ATSC2dv', 'ATSC1p',
    'AATSC1v', 'AATSC5v', 'AATSC1i', 'MATS2s', 'MATS3Z', 'MATS5Z', 'GATS8c', 'GATS2s',
    'GATS3Z', 'GATS5Z', 'GATS4v', 'GATS1se', 'BCUTdv-1h', 'BCUTv-1l', 'nBondsT', 'HybRatio',
    'Xc-3d', 'AXp-6d', 'AXp-6dv', 'NsCH3', 'NsssCH', 'NssO', 'AETA_eta_L', 'fMF',
    'nHBDon', 'IC1', 'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA10', 'MDEC-13', 'RotRatio', 'TopoPSA(NO)']


    # 选择需要的描述符
    calc_all = Calculator(descriptors, ignore_3D=True)
    selected_descriptors = [d for d in calc_all.descriptors if str(d) in FEATURES_USED]
    calc = Calculator(selected_descriptors, ignore_3D=True)

    # 构建标准化器
    df_x = pd.read_csv(os.path.join(MODULE_DIR, "df_final_DM.csv"))
    X_all = df_x[FEATURES_USED].values
    scaler = StandardScaler()
    X_scaled_all = scaler.fit(X_all)

    # 分子解析
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("无效的 SMILES：无法解析为分子对象")
    
    try:
        desc = calc(mol)
        desc_values = np.array([desc[str(d)] for d in calc.descriptors], dtype=float).reshape(1, -1)
    except Exception as e:
        raise ValueError(f"描述符计算失败：{e}")
    
    # 处理 NaN：将所有 NaN 替换为 0
    if np.isnan(desc_values).any():
        desc_values = np.nan_to_num(desc_values, nan=0.0)

    # 标准化 + 预测
    desc_scaled = scaler.transform(desc_values)
    prediction = DM_model.predict(desc_scaled)
    return float(prediction)

# print(activity_func('N#Cc1ccc(SCC(=O)N2CCC(c3ccccc3)C2)cc1'))
# print(model.jtnn_vae.encode_from_smiles(['N#Cc1ccc(SCC(=O)N2CCC(c3ccccc3)C2)cc1']))

