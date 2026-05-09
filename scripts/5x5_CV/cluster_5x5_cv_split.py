import os
import random
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from tqdm import tqdm

# Disable Pandas chained_assignment warning
pd.options.mode.chained_assignment = None 

# ==========================================
# Core function: Generate 5x5 Butina clustering cross-validation data
# ==========================================
def generate_5x5_cluster_splits(df, data_name, cutoff=0.4, seeds=[0, 1, 2, 3, 4]):
    # 1. Calculate molecular fingerprints
    print("Step 1: Calculating molecular fingerprints...")
    mols = []
    for s in tqdm(df['smiles'], desc="Parsing SMILES"):
        mols.append(Chem.MolFromSmiles(s))
    
    # Remove invalid molecules
    valid_idxs = [i for i, m in enumerate(mols) if m is not None]
    if len(valid_idxs) < len(df):
        print(f"⚠️ Removed {len(df)-len(valid_idxs)} invalid molecules")
        df = df.iloc[valid_idxs].reset_index(drop=True)
        mols = [m for m in mols if m is not None]

    # Generate Morgan fingerprints (Radius 2, 1024 bits)
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]
    
    # 2. Calculate distance matrix (1 - Tanimoto similarity)
    print("Step 2: Calculating distance matrix...")
    dists = []
    n_fps = len(fps)
    
    # RDKit's clustering function requires a lower-triangular distance matrix
    for i in tqdm(range(1, n_fps), desc="Calculating Distance Matrix"):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    
    # 3. Perform a single clustering
    print(f"Step 3: Performing Butina clustering (cutoff={cutoff})...")
    clusters = Butina.ClusterData(dists, n_fps, cutoff, isDistData=True)
    # clusters is a tuple of tuples, where each inner tuple contains the molecule indices belonging to that cluster
    print(f"  -> Generated a total of {len(clusters)} clusters")

    # Base save path
    base_save_dir = f'./processed_datasets/cluster_5x5_cv/{data_name}'

    # 4. Start 5 repeated iterations (5 Seeds)
    for seed in seeds:
        print(f"\n>>> Processing random seed: {seed} <<<")
        random.seed(seed)
        np.random.seed(seed)
        
        # Convert clusters to a list so we can shuffle their order
        cluster_list = list(clusters)
        random.shuffle(cluster_list)
        
        # Initialize 5 empty folds
        folds = [[] for _ in range(5)]
        
        # Randomized greedy assignment: put each cluster into the fold with the fewest molecules currently
        # Ensure molecules belonging to the same cluster are never split across folds
        for cluster_indices in cluster_list:
            min_fold_idx = np.argmin([len(f) for f in folds])
            folds[min_fold_idx].extend(cluster_indices)
            
        # Print the fold size distribution for the current seed
        fold_sizes = [len(f) for f in folds]
        print(f"  5-fold size distribution: {fold_sizes} (Total: {sum(fold_sizes)})")
        
        # 5. Start 5-fold cross-validation splitting (5 Folds)
        for fold_idx in range(5):
            # Setting: current fold_idx is the test set, the next fold is the validation set, and the remaining 3 are the training set
            test_idx = fold_idx
            valid_idx = (fold_idx + 1) % 5
            train_idxs = [i for i in range(5) if i != test_idx and i != valid_idx]
            
            # Extract the actual indices of the corresponding molecules in the current dataframe
            test_indices = folds[test_idx]
            valid_indices = folds[valid_idx]
            train_indices = []
            for t_i in train_idxs:
                train_indices.extend(folds[t_i])
                
            # Extract data via iloc (since we reset_index at the beginning, we can use integer indexing directly)
            train_df = df.iloc[train_indices]
            valid_df = df.iloc[valid_indices]
            test_df  = df.iloc[test_indices]
            
            # Create save directory: seed_X/fold_Y/
            save_dir = os.path.join(base_save_dir, f'seed_{seed}', f'fold_{fold_idx}')
            os.makedirs(save_dir, exist_ok=True)
            
            train_df.to_csv(os.path.join(save_dir, 'train.csv'), index=False)
            valid_df.to_csv(os.path.join(save_dir, 'valid.csv'), index=False)
            test_df.to_csv(os.path.join(save_dir, 'test.csv'), index=False)
            
            if fold_idx == 0: # Print Fold 0 example
                print(f"  [Fold 0 Example] Train: {len(train_df)} | Valid: {len(valid_df)} | Test: {len(test_df)}")

    print(f"\n✅ 5x5 Butina clustering cross-validation generation complete! Saved at: {base_save_dir}")

# ==========================================
# Main program
# ==========================================
if __name__ == "__main__":

    property_name_list = ["Egc", "Egb", "Eea", "Ei", "Xc", "EPS", "Nc", "Eat"]

    for property_name in property_name_list:

        print(f"\n{'='*10} [5x5 Cluster CV] Processing dataset: {property_name} {'='*10}")
        
        csv_path = f'../../datasets/raw_datasets/{property_name}.csv'
        
        df = pd.read_csv(csv_path)
        df.columns = ['smiles', 'label'] + list(df.columns[2:])
        generate_5x5_cluster_splits(df, property_name, cutoff=0.4)

   