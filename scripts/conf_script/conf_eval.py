import os
import csv
import sys
import torch
import numpy as np
import pandas as pd
from pymol import cmd
from tqdm import tqdm
from rdkit import Chem
from pathlib import Path
from copy import deepcopy
from rdkit import RDLogger
from rdkit.Chem import rdchem
from rdkit.Chem import AllChem  
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from openfold.utils.rigid_utils import Rigid
RDLogger.DisableLog('rdApp.*')


def process_monomer(monomer_smiles):
    smi = monomer_smiles
    mol = Chem.MolFromSmiles(smi)
    
    assert 'H' not in [atom.GetSymbol() for atom in mol.GetAtoms()]
    
    key_point_list = []
    for atom in mol.GetAtoms():
        atom_neighbors = atom.GetNeighbors()
        for atom_neighbor in atom_neighbors:
            if atom_neighbor.GetSymbol() == '*':
                key_point_list.append(atom.GetIdx())
                key_point_list.append(atom_neighbor.GetIdx())
    
    assert len(key_point_list) == 4, 'Unvalid PSMILES'

    star_0 = key_point_list[1]
    star_1 = key_point_list[3]
    neighbor_0 = key_point_list[0]
    neighbor_1 = key_point_list[2]
    
    atom = mol.GetAtomWithIdx(star_0)
    atom.SetAtomicNum(mol.GetAtomWithIdx(neighbor_1).GetAtomicNum())

    atom = mol.GetAtomWithIdx(star_1)
    atom.SetAtomicNum(mol.GetAtomWithIdx(neighbor_0).GetAtomicNum())

    processed_smi = ""
    replacement = [str(mol.GetAtomWithIdx(neighbor_1).GetSymbol()), str(mol.GetAtomWithIdx(neighbor_0).GetSymbol())]
    count = 0
    i = 0
    while i != len(smi):
        if smi[i:i+3] == "[*]":
            processed_smi += replacement[count]
            count += 1
            i += 3
        else:
            processed_smi += smi[i]
            i += 1

    pre_atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    processed_mol = Chem.MolFromSmiles(processed_smi)
    post_atoms = [atom.GetSymbol() for atom in processed_mol.GetAtoms()]
    
    assert pre_atoms == post_atoms, 'Unmatch Order'
    
    return key_point_list, processed_mol


def rot_mat_mul(rots, mats):    
    mats = mats.unsqueeze(-1)  
    rots = rots.unsqueeze(1) 
    rotated_mats = torch.matmul(rots, mats)  
    return rotated_mats.squeeze(-1)  


def extract_mol_info(mol, key_point_index):
    emol = Chem.RWMol(deepcopy(mol))
    emol.RemoveAtom(key_point_index[1])
    if key_point_index[3] > key_point_index[1]:
        emol.RemoveAtom(key_point_index[3] - 1)
    else:
        emol.RemoveAtom(key_point_index[3])
    processed_mol = emol.GetMol()  
    
    prior_idx = key_point_index[2]
    post_idx = key_point_index[0]
    if prior_idx > key_point_index[1]:
        prior_idx = prior_idx - 1
    if prior_idx > key_point_index[3]:
        prior_idx = prior_idx - 1
    if post_idx > key_point_index[1]:
        post_idx = post_idx - 1
    if post_idx > key_point_index[3]:
        post_idx = post_idx - 1
    
    bond_type = mol.GetBondBetweenAtoms(key_point_index[2], key_point_index[3]).GetBondType()
    return processed_mol, prior_idx, post_idx, bond_type
    

def extract_head_mol_info(mol, key_point_index):
    head_emol = Chem.RWMol(deepcopy(mol))
    head_emol.RemoveAtom(key_point_index[3])
    head_processed_mol = head_emol.GetMol()  
    head_prior_idx = key_point_index[2]
    if head_prior_idx > key_point_index[3]:
        head_prior_idx = head_prior_idx - 1
    return head_processed_mol, head_prior_idx


def extract_tail_mol_info(mol, key_point_index):
    tail_emol = Chem.RWMol(deepcopy(mol))
    tail_emol.RemoveAtom(key_point_index[1])
    tail_processed_mol = tail_emol.GetMol()  
    tail_post_idx = key_point_index[0]
    if tail_post_idx > key_point_index[1]:
        tail_post_idx = tail_post_idx - 1
    return tail_processed_mol, tail_post_idx


def get_filtered_mol(mol, R_h):
    mol_h = deepcopy(mol)
    R_h = R_h.tolist()
    conf_h = rdchem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf_h.SetAtomPosition(i, R_h[i])
    mol_h.RemoveConformer(0)
    mol_h.AddConformer(conf_h)
    return mol_h


def get_filtered_mols(mol, filtered_confs):
    filtered_mols = []
    for idx in range(filtered_confs.shape[0]):
        filtered_mol = get_filtered_mol(mol, filtered_confs[idx])
        filtered_mols.append(filtered_mol)
    return filtered_mols
    

def get_merged_mol(merged_mol, next_mol, prior_idx, post_idx, bond_type):        
    emol1 = Chem.RWMol(deepcopy(merged_mol))  
    emol2 = Chem.RWMol(deepcopy(next_mol))  
        
    combined = Chem.CombineMols(emol1, emol2)  
    rw_combine = Chem.RWMol(combined)  
    rw_combine.AddBond(prior_idx, emol1.GetNumAtoms() + post_idx, bond_type)
        
    merged_mol = rw_combine.GetMol()  
    return merged_mol
    

def merge_mols(filtered_mols, prior_idx, post_idx, bond_type):
    merged_mol = deepcopy(filtered_mols[0])
    offset = filtered_mols[0].GetNumAtoms()
    for idx in range(1, len(filtered_mols)):
        merged_mol = get_merged_mol(merged_mol, filtered_mols[idx], prior_idx, post_idx, bond_type)
        prior_idx = prior_idx + offset
    return merged_mol, prior_idx
        
    
def write_molecule_to_sdf(molecule, sdf_filename):
    writer = Chem.SDWriter(sdf_filename)
    writer.SetKekulize(False)
    writer.write(molecule)
    writer.close()

    
def apply_frames_confs(psmi, local_frames, local_confs):
    key_point_index, mol = process_monomer(psmi)
    key_point_confs = local_confs[:, key_point_index]
          
    base_atom_pos = torch.from_numpy(np.concatenate((key_point_confs[0:1][:, [0, 2, 3], :], key_point_confs[1:-1][:, [1, 0, 3], :], key_point_confs[-1:][:, [1, 0, 2], :]), axis=0))
    gt_rigids = Rigid.from_3_points(
        p_neg_x_axis=base_atom_pos[..., 0, :],
        origin=base_atom_pos[..., 1, :],
        p_xy_plane=base_atom_pos[..., 2, :],
        eps=1e-8,
    )
    stand_confs = rot_mat_mul(gt_rigids._rots.get_rot_mats().transpose(-1, -2), torch.from_numpy(local_confs))
        
    local_rigids = Rigid.from_tensor_7(torch.from_numpy(local_frames))
    rot_confs = rot_mat_mul(local_rigids._rots.get_rot_mats(), stand_confs) 
    
    rot_key_point_confs = rot_confs.numpy()[:, key_point_index]
    trans_confs = np.cumsum(np.vstack((np.zeros((1, 3)), (rot_key_point_confs[:-1, 3] - rot_key_point_confs[1:, 0]))), axis=0)  
    processed_confs = rot_confs + torch.from_numpy(trans_confs).unsqueeze(-2)
   
    processed_mol, prior_idx, post_idx, bond_type = extract_mol_info(mol, key_point_index)
    indices_to_keep = [i for i in range(processed_confs.shape[1]) if (i != key_point_index[1] and i != key_point_index[3])]
    filtered_confs = processed_confs[1:-1, indices_to_keep, :].numpy() 
    filtered_mols = get_filtered_mols(processed_mol, filtered_confs)
    merged_mol, merged_prior_idx = merge_mols(filtered_mols, prior_idx, post_idx, bond_type)
        
    tail_processed_mol, tail_post_idx = extract_tail_mol_info(mol, key_point_index)
    tail_indices_to_keep = [i for i in range(processed_confs.shape[1]) if (i != key_point_index[1])]
    tail_filtered_confs = processed_confs[-1:, tail_indices_to_keep, :].numpy() 
    tail_filtered_mol = get_filtered_mol(tail_processed_mol, tail_filtered_confs[0])
    merged_mol = get_merged_mol(merged_mol, tail_filtered_mol, merged_prior_idx, tail_post_idx, bond_type)

    head_processed_mol, head_prior_idx = extract_head_mol_info(mol, key_point_index)
    head_indices_to_keep = [i for i in range(processed_confs.shape[1]) if (i != key_point_index[3])]
    head_filtered_confs = processed_confs[:1, head_indices_to_keep, :].numpy() 
    head_filtered_mol = get_filtered_mol(head_processed_mol, head_filtered_confs[0])
    merged_mol = get_merged_mol(head_filtered_mol, merged_mol, head_prior_idx, post_idx, bond_type)

    write_molecule_to_sdf(merged_mol, os.path.join(fold_path, f'generated_conf_{gen_idx - 1}.sdf'))
    
   
def extract_confs(local_mols):
    local_confs_dict = {}
    for idx, local_mol in enumerate(local_mols):
        local_mol = Chem.RemoveHs(local_mol)
        local_confs_dict[idx] = local_mol.GetConformer().GetPositions().astype(np.float32)
    local_confs = np.array([local_confs_dict[idx] for idx in local_confs_dict.keys()])
    return local_confs


def cal_energy(mol_path):
    mol = Chem.MolFromMolFile(mol_path, removeHs=False)
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)  
    mmff_forcefield = AllChem.MMFFGetMoleculeForceField(mol, mmff_props)  
    mmff_energy = (mmff_forcefield.CalcEnergy()) /  mol.GetNumAtoms() 
    return mmff_energy


def eval(gen_path, refer_path):
    s_matr_scores = []
    s_matp_scores = []
    e_matr_scores = []
    e_matp_scores = []
    
    data_idx_list = [gen_mol_path.split('_')[1] for gen_mol_path in os.listdir(gen_path)]

    for data_idx in tqdm(data_idx_list):
        rmsd_mat = -1 * np.ones([5, 10], dtype=float)
        energy_mat = -1 * np.ones([5, 10], dtype=float)
        for i in range(10):
            for j in range(5):
                gen_mol_path = os.path.join(gen_path, f'data_{data_idx}', f'generated_conf_{i}.sdf')
                true_mol_path = os.path.join(refer_path, f'data_{data_idx}', f'true_conf_{j}.sdf')
                cmd.reinitialize()
                cmd.feedback("disable", "matrix", "everything")
                cmd.load(true_mol_path, 'true_mol')
                cmd.load(gen_mol_path, 'gen_mol')
                rmsd = cmd.align('true_mol', 'gen_mol')[0]
                rmsd_mat[j,i] = rmsd
                energy_mat[j,i] = (cal_energy(gen_mol_path)-cal_energy(true_mol_path)) * 0.0015936

        rmsd_ref_min = rmsd_mat.min(-1)    
        rmsd_gen_min = rmsd_mat.min(0)
        s_matr_scores.append(rmsd_ref_min.mean())
        s_matp_scores.append(rmsd_gen_min.mean())

        energy_ref_min = energy_mat.min(-1)    
        energy_gen_min = energy_mat.min(0)
        e_matr_scores.append(energy_ref_min.mean())
        e_matp_scores.append(energy_gen_min.mean())

    s_matr_scores = np.array(s_matr_scores)
    s_matp_scores = np.array(s_matp_scores)
    print('S-MAT-R_mean: %.4f | S-MAT-R_median: %.4f' % (np.mean(s_matr_scores), np.median(s_matr_scores)))
    print('S-MAT-P_mean: %.4f | S-MAT-P_median: %.4f' % (np.mean(s_matp_scores), np.median(s_matp_scores)))

    e_matr_scores = np.array(e_matr_scores)
    e_matp_scores = np.array(e_matp_scores)
    print('E-MAT-R_mean: %.4f | E-MAT-R_median: %.4f' % (np.mean(e_matr_scores), np.median(e_matr_scores)))
    print('E-MAT-P_mean: %.4f | E-MAT-P_median: %.4f' % (np.mean(e_matp_scores), np.median(e_matp_scores)))


if __name__ == '__main__':
    for gen_idx in range(1, 11):
        refer_path = './datasets/pretrain_dataset/test_data_index.csv'
        generated_path = f'./results/conf_result/conf_result_{gen_idx}/test.out.pkl'
        
        local_mols_info = {k: v['mols'] for d in pd.read_pickle(generated_path) for k, v in d.items() if isinstance(v, dict)}
        local_frames_info = {k: v['rigids'] for d in pd.read_pickle(generated_path) for k, v in d.items() if isinstance(v, dict)}
        
        with open(refer_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in tqdm(reader):
                file_id = row[0]
                psmi = row[1]
                if psmi in local_mols_info.keys():
                    local_confs = extract_confs(local_mols_info[psmi])
                    fold_path = f'./results/conf_result/generated_confs/data_{file_id}'
                    os.makedirs(fold_path, exist_ok=True)
                    apply_frames_confs(psmi, local_frames_info[psmi][0], local_confs)

    eval('./results/conf_result/generated_confs', './datasets/pretrain_dataset/true_confs')
   