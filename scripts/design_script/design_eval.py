import os
import sys
import json
import torch
import pickle
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from polyconfm.models.design_module.diffusion.distributions import DistributionNodes
from polyconfm.models.design_module.metrics.molecular_metrics_sampling import SamplingMolecularMetrics


class DataInfos():
    def __init__(self, data_path, task_name, ensure_connected):
        tasktype_dict = {
            'O2': 'regression',
            'N2': 'regression',
            'CO2': 'regression',
        }
        self.task = task_name
        self.task_type = tasktype_dict.get(task_name, "regression")
        self.ensure_connected = ensure_connected

        meta_filename = os.path.join(data_path, f'{task_name}.meta.json')
        with open(meta_filename, 'r') as f:
            meta_dict = json.load(f)

        self.base_path = data_path
        self.active_atoms = meta_dict['active_atoms']
        self.max_n_nodes = meta_dict['max_node']
        self.original_max_n_nodes = meta_dict['max_node']
        self.n_nodes = torch.Tensor(meta_dict['n_atoms_per_mol_dist'])
        self.edge_types = torch.Tensor(meta_dict['bond_type_dist'])
        self.transition_E = torch.Tensor(meta_dict['transition_E'])

        self.atom_decoder = meta_dict['active_atoms']
        node_types = torch.Tensor(meta_dict['atom_type_dist'])
        active_index = (node_types > 0).nonzero().squeeze()
        self.node_types = torch.Tensor(meta_dict['atom_type_dist'])[active_index]
        self.nodes_dist = DistributionNodes(self.n_nodes)
        self.active_index = active_index

        val_len = 3 * self.original_max_n_nodes - 2
        meta_val = torch.Tensor(meta_dict['valencies'])
        self.valency_distribution = torch.zeros(val_len)
        val_len = min(val_len, len(meta_val))
        self.valency_distribution[:val_len] = meta_val[:val_len]
        self.y_prior = None
        self.train_ymin = []
        self.train_ymax = []

        self.input_dims = {'X': len(self.active_index), 
                           'E': 5, 
                           'y': 2 + len(task_name.split('-'))}
        self.output_dims = {'X': len(self.active_index),
                            'E': 5,
                            'y': 2 + len(task_name.split('-'))}
        

def get_psmi_list(data_path, split_name):
    with open(os.path.join(data_path, f'{split_name}_psmi.pkl'), 'rb') as file:  
        psmi_list = list(pickle.load(file))
    psmi_list = [Chem.MolToSmiles(Chem.MolFromSmiles(psmi)) for psmi in psmi_list]
    return psmi_list


if __name__ == '__main__':

    root_path = f'./results/design_result'
    input_path = f'./results/design_result/test.out.pkl'

    samples, all_ys = [], []
    for d in tqdm(pd.read_pickle(input_path)):
        samples = samples + d['sample']
        all_ys.append(d['batch_y'])
    
    dataset_infos = DataInfos('./datasets/design_dataset', 'O2-N2-CO2', True)
    train_smiles = get_psmi_list('./datasets/design_dataset', 'train')
    reference_smiles = get_psmi_list('./datasets/design_dataset', 'test')
    sampling_metrics = SamplingMolecularMetrics(
            dataset_infos, train_smiles, reference_smiles
        )
    sampling_metrics.reset()
    sampling_metrics(root_path, samples, all_ys, 'polyconf', 1, 1, test=True)
    sampling_metrics.reset()