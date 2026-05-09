import os
import random
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

# Disable Pandas chained_assignment warning
pd.options.mode.chained_assignment = None 

# ==========================================
# Helper functions
# ==========================================
def generate_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except:
        return None
    return None

# ==========================================
# Core modification: Generate 5x5 scaffold cross-validation data
# ==========================================
def generate_5x5_scaffold_splits(df, data_name, seeds=[0, 1, 2, 3, 4]):
    # 1. Generate scaffolds
    tqdm.pandas(desc="Generating Scaffolds")
    df['scaffold'] = df['smiles'].progress_apply(generate_scaffold)
    
    # Remove invalid molecules
    failed_count = df['scaffold'].isnull().sum()
    if failed_count > 0:
        print(f"⚠️ Warning: Found {failed_count} molecules that failed to generate scaffolds, they have been removed.")
        df = df.dropna(subset=['scaffold']).reset_index(drop=True)
    
    if len(df) == 0:
        raise ValueError("Scaffold generation failed for all molecules in the dataset!")

    # 2. Pack indices by scaffold (bundle homologous molecules)
    scaffold_sets = {}
    for idx, row in df.iterrows():
        s = row['scaffold']
        if s not in scaffold_sets: 
            scaffold_sets[s] = []
        scaffold_sets[s].append(idx)
        
    scaffolds = list(scaffold_sets.keys())
    
    # Base save path
    base_save_dir = f'./processed_datasets/scaffold_5x5_cv/{data_name}'
    
    # 3. Start 5 repeated iterations (5 Seeds)
    for seed in seeds:
        print(f"\n>>> Processing random seed: {seed} <<<")
        random.seed(seed)
        np.random.seed(seed)
        
        # Shuffle the scaffold list (ensure the 5 folds generated each time are completely different)
        random.shuffle(scaffolds)
        
        # Initialize 5 empty folds
        folds = [[] for _ in range(5)]
        
        # Randomized greedy assignment: put each scaffold into the fold with the fewest molecules currently
        # This ensures scaffolds don't cross folds, and keeps the sizes of the 5 folds as close as possible
        for s in scaffolds:
            min_fold_idx = np.argmin([len(f) for f in folds])
            folds[min_fold_idx].extend(scaffold_sets[s])
            
        # Print the fold size distribution for the current seed
        fold_sizes = [len(f) for f in folds]
        print(f"  5-fold size distribution: {fold_sizes} (Total: {sum(fold_sizes)})")
        
        # 4. Start 5-fold cross-validation splitting (5 Folds)
        for fold_idx in range(5):
            # Setting: current fold_idx is the test set, the next fold is the validation set, and the remaining 3 are the training set
            # This rotation ensures all data serves exactly once as the test set
            test_idx = fold_idx
            valid_idx = (fold_idx + 1) % 5
            train_idxs = [i for i in range(5) if i != test_idx and i != valid_idx]
            
            # Extract the corresponding molecule indices
            test_indices = folds[test_idx]
            valid_indices = folds[valid_idx]
            train_indices = []
            for t_i in train_idxs:
                train_indices.extend(folds[t_i])
                
            # Extract data from DataFrame and drop the scaffold column
            train_df = df.loc[train_indices].drop(columns=['scaffold'])
            valid_df = df.loc[valid_indices].drop(columns=['scaffold'])
            test_df  = df.loc[test_indices].drop(columns=['scaffold'])
            
            # Create save directory: seed_X/fold_Y/
            save_dir = os.path.join(base_save_dir, f'seed_{seed}', f'fold_{fold_idx}')
            os.makedirs(save_dir, exist_ok=True)
            
            train_df.to_csv(os.path.join(save_dir, 'train.csv'), index=False)
            valid_df.to_csv(os.path.join(save_dir, 'valid.csv'), index=False)
            test_df.to_csv(os.path.join(save_dir, 'test.csv'), index=False)
            
            if fold_idx == 0: # Only print Fold 0 example for each seed to avoid flooding the screen
                print(f"  [Fold 0 Example] Train: {len(train_df)} | Valid: {len(valid_df)} | Test: {len(test_df)}")

    print(f"\n✅ 5x5 cross-validation dataset generation complete! A total of 25 splits generated, saved at: {base_save_dir}")


if __name__ == "__main__":

    property_name_list = ["Egc", "Egb", "Eea", "Ei", "Xc", "EPS", "Nc", "Eat"]

    for property_name in property_name_list:

        print(f"\n{'='*10} [5x5 Scaffold CV] Processing dataset: {property_name} {'='*10}")
        
        csv_path = f'../../datasets/raw_datasets/{property_name}.csv'
        
        df = pd.read_csv(csv_path)
        df.columns = ['smiles', 'label'] + list(df.columns[2:])
        generate_5x5_scaffold_splits(df, property_name)