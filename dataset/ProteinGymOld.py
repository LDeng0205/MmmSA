import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from tqdm import tqdm
from collections import OrderedDict

from utils.blosum62 import compute_blosum62_score

import os
import glob

class SubsampleDataset(Dataset):
    def __init__(self, original_dataset, balance=False, balance_cutoff=0.14, balance_feature=None, num_samples=1000):
        pass
    def __len__(self):
        pass
    def __getitem__(self, idx):
        pass
    def get_LA(self):
        pass
    def get_dms(self):
        pass

class ProteinGym(Dataset):
    def __init__(self, name, alphabet=None, balance=False, features=None, domain_only=False,
                 orig_weights_cutoffs=None, normalization='standard', pc_path=None, random_permute=False,
                 esm_embeddings_path='/home/hanlunj/projects/protein_gym/20231011_esm_embeddings',
                 data_dir='/home/arthur/data/ProteinGym2024'):
        if alphabet is None:
            self.alphabet = '-ACDEFGHIKLMNPQRSTVWY'
        else:
            self.alphabet = alphabet
        self.data_dir = data_dir
        df = pd.read_csv('functional_sites.csv')
        # print(name)
        dataset_info = df[df['dataset'] == name]

        df = pd.read_csv(os.path.join(self.data_dir, 
                                      "Tranception/proteingym/Detailed_performance_files/Substitutions" \
                                        "/Spearman/all_models_substitutions_Spearman_DMS_level.csv"))
        df = df.rename(columns={df.columns[0]: 'Dataset'})
        df = df.loc[df.EVmutation.sort_values(ascending=False).index, ["Dataset", "EVmutation",  "EVE_single", "EVE_ensemble","UniProt_ID", "Neff_L_category", "Taxon"]]
        msa_fn, weights, res_df, mut_seqs = self.get_proteingym_data(name)
        
        # Process MSA sequences
        msa_sequences_raw = [str(x.seq) for x in SeqIO.parse(msa_fn, 'fasta')]
        msa_sequences_raw = [s.replace(".", "-").upper() for s in msa_sequences_raw]
        wt_seq = msa_sequences_raw[0]
        self.wt_raw = wt_seq
        
        columns_to_keep = [i for i in range(len(wt_seq))]

        msa_sequences_raw = [s for s in msa_sequences_raw if self.check_sequence(s)]

        msa_sequences = [[s[i] for i in columns_to_keep] for s in msa_sequences_raw]
        msa_sequences = np.asarray(msa_sequences)

        if domain_only:
            domain_range = dataset_info['domain_range'].values[0]
            self.domain_range = tuple(map(lambda x: int(x)-1, domain_range.split('-')))
            print(f'Only keeping positions in the domain pos {self.domain_range[0]} to {self.domain_range[1]}')
            msa_sequences = [s[self.domain_range[0]:self.domain_range[1]] for s in msa_sequences]
            mut_seqs = mut_seqs[:, self.domain_range[0]:self.domain_range[1]]

        seqs_enc, aa_to_i = self.encode(msa_sequences)

        i_to_a = {i:aa for i, aa in enumerate(self.alphabet)}

        assert weights.shape[0] == len(msa_sequences)

        seqs = torch.nn.functional.one_hot(seqs_enc, num_classes=len(self.alphabet)).to(torch.float32)
        
        # Process DMS sequences
        mut_seqs_onehot = torch.nn.functional.one_hot(mut_seqs, num_classes=len(self.alphabet)).to(torch.float32)
        y_dms = res_df.DMS_score.to_numpy() # get ground truth DMS data

        self.seqs = seqs
        
        self.name = name
        self.balance = balance

        if domain_only:
            # recompute weights
            print('Recomputed clustered weights.')
            self.orig_weights = compute_weights(self.seqs, theta=0.2).cpu()
        else:
            print('Using weights from protein gym.')
            self.orig_weights = torch.tensor(weights)
        self.features = features
        self.data = torch.tensor([])
        
        if not self.features is None:
            # featurization requiring raw sequence strings
            for feature in self.features:
                if feature == 'functional_sites':
                    functional_sites = dataset_info['functional_sites'].values[0]
                    calc_func_idx = lambda x: int(x)-1-(self.domain_range[0] if domain_only else 0)# convert functional sites to sequence indices
                    self.functional_sites = list(map(calc_func_idx, functional_sites.split('_')))
                    print('Functional sites indices: ', self.functional_sites)
                    f_sites_hamming = torch.tensor([(np.array(list(s))[self.functional_sites] != np.array(list(wt_seq))[self.functional_sites]).sum() 
                                                                       for s in msa_sequences_raw]).unsqueeze(-1)
                    self.data = torch.hstack((self.data, f_sites_hamming))
                elif feature == 'gap':
                    self.data = torch.hstack((self.data, torch.tensor([s.count('-') for s in msa_sequences_raw], dtype=torch.float32).unsqueeze(-1)))
                elif feature == 'blosum':
                    self.data = torch.hstack((self.data, torch.tensor([compute_blosum62_score(s, wt_seq) for s in msa_sequences_raw], dtype=torch.float32).unsqueeze(-1)))
                elif feature == 'hamming':
                    f_func = [lambda s : (s != self.seqs[0]).sum().item()]
                    self.data = torch.hstack((self.data, torch.tensor(self.compute_features(self.seqs, f_func))))
                elif feature == 'hamming_square':
                    f_func = [lambda s : (s != self.seqs[0]).sum().item() ** 2]
                    self.data = torch.hstack((self.data, torch.tensor(self.compute_features(self.seqs, f_func))))
                elif feature == 'orig_weights':
                    cutoffs = [0.2] if orig_weights_cutoffs is None else orig_weights_cutoffs 
                    orig_weights = [compute_weights(self.seqs, theta=cutoff).cpu().unsqueeze(-1) for cutoff in cutoffs]
                    for orig_w in orig_weights:
                        self.data = torch.hstack((self.data, orig_w))
                elif feature == 'pcavar':
                    f_func = [lambda s : (s != self.seqs[0]).sum().item()]
                    ham_dist = torch.tensor(self.compute_features(self.seqs, f_func))
                    _, idx = torch.sort(ham_dist)
                    sorted_seqs = self.seqs[idx].flatten(1).cuda()
                    u, s, vt = torch.pca_lowrank(sorted_seqs)
                    # Use 5 principal components
                    projected = sorted_seqs@vt[:, :5]

                    vars = []
                    print('Computing variance feature.')
                    for i in tqdm(range(projected.shape[0])):
                        vars.append(average_empirical_variance(projected[:i]).item())
                    vars = torch.tensor(vars)
                    feature = torch.empty_like(vars)
                    print(vars.shape)
                    print(feature.shape)
                    feature[idx] = vars.unsqueeze(1)
                    self.data = torch.hstack((self.data, feature.unsqueeze(-1)))
                elif feature == 'esm_l1':
                    embeddings = torch.load(f'{esm_embeddings_path}/{name}.esm2_t33_650M_UR50D.pt')
                    l1_distances = torch.sum(torch.abs(embeddings - embeddings[0]), dim=1)
                    self.data = torch.hstack((self.data, l1_distances.unsqueeze(-1)))
                elif feature == 'esm_l2':
                    embeddings = torch.load(f'{esm_embeddings_path}/{name}.esm2_t33_650M_UR50D.pt')
                    l2_distances = torch.norm(embeddings - embeddings[0], dim=1)
                    self.data = torch.hstack((self.data, l2_distances.unsqueeze(-1)))
                elif feature == 'esm_dot':
                    embeddings = torch.load(f'{esm_embeddings_path}/{name}.esm2_t33_650M_UR50D.pt')
                    dot_prod = embeddings @ embeddings[0]
                    self.data = torch.hstack((self.data, dot_prod.unsqueeze(-1)))
                elif feature == 'esm_pca':
                    embeddings = torch.load(f'{esm_embeddings_path}/{name}.esm2_t33_650M_UR50D.pt')
                    pc = torch.load(pc_path)
                    projected_emb = embeddings @ pc
                    print(f'Using projected embeddings of shape {projected_emb.shape}')
                    self.data = torch.hstack((self.data, projected_emb))
                else:
                    raise ValueError('Error: Unrecognized feature: ', feature)
            
            self.data = self.data.to(torch.float32)
            self.data_pre_norm = self.data.clone()

            # normalize features
            if normalization == 'standard':
                mean = torch.mean(self.data, dim=0)
                std = torch.std(self.data, dim=0)
                self.data = (self.data - mean) / (std + 1e-8)
            elif normalization == 'percentile':
                self.data = tensor_to_percentiles(self.data)
            else:
                print("No normalization method is used.")
            print(f'Constructed dataset with features: {self.features}.')
            print(f'Data shape: {self.data.shape}')

        # self.data = torch.hstack((self.data, self.orig_weights.unsqueeze(-1).to(torch.float32))) # Add original weights as a feature.
        if random_permute:
            print('Randomly permuting the features and the sequences for ablations')
            perm = torch.randperm(self.data.size(0))
            self.data = self.data[perm]

        self.dms_X = mut_seqs_onehot
        self.dms_y = torch.tensor(y_dms)
        self.wt_enc = seqs[0].argmax(dim=-1)
        self.L = len(seqs_enc[0])
        self.A = len(self.alphabet)
        self.eff_seq = np.sum(weights) # Number of "effective sequences," sum of default weights

    def __len__(self):
        return self.seqs.shape[0]
    
    def __getitem__(self, idx):
        return self.seqs[idx], self.data[idx], self.orig_weights[idx]
    
    def get_dms(self):
        return self.dms_X, self.dms_y
    
    def get_LA(self):
        return self.L, self.A
    
    def check_sequence(self, s):
        for aa in s:
            if aa not in self.alphabet:
                return False
        return True
    
    def encode(self, seqs, verbose=True):
        '''
        Go from letters to numbers
        '''
        aa_to_i = OrderedDict((aa, i) for i, aa in enumerate( self.alphabet ))
        if verbose:
            seq_iter = tqdm(seqs)
        else:
            seq_iter = seqs
        X = torch.tensor([[aa_to_i[x] for x in seq] 
                          for seq in seq_iter])
        return X, aa_to_i
    
    def compute_features(self, msa_seq, featurize_funcs):
        featurized_data = []
        
        for s in msa_seq:
            featurized_sequence = []
            for f in featurize_funcs:
                featurized_sequence.append(f(s))
            featurized_data.append(featurized_sequence)

        return featurized_data
        
    def get_proteingym_data(self, dataset):
        aa_to_i = {aa: i for i, aa in enumerate(self.alphabet)}

        msa_files = glob.glob(os.path.join(self.data_dir, 'MSA_files/*.a2m'))
        mutation_files = glob.glob(os.path.join(self.data_dir, 'ProteinGym_substitutions/*.csv'))
        weight_files = glob.glob(os.path.join(self.data_dir, 'substitutions_MSAs_all_positions/*.npy'))
        # res_files = glob.glob(os.path.join(self.data_dir, 'substitutions/*.csv'))

        msa_matches = [f for f in msa_files if dataset.split("_")[0] in os.path.basename(f)]
        if len(msa_matches) > 1:
            msa_matches = [
                f for f in msa_files if "_".join(dataset.split("_")[:2]) in os.path.basename(f)
            ]

        assert len(msa_matches) == 1

        mut_matches = [f for f in mutation_files if dataset in os.path.basename(f)]

        assert len(mut_matches) == 1
        weight_matches = [
            f for f in weight_files if dataset.split("_")[0] in os.path.basename(f)
        ]
        weight_matches = [
            f for f in weight_files if dataset.split("_")[0] in os.path.basename(f)
        ]
        if len(weight_matches) > 1:
            weight_matches = [
                f for f in weight_files
                if "_".join(dataset.split("_")[:2]) in os.path.basename(f)
            ]
        if dataset == 'A4_HUMAN_Seuma_2021':
            weight_matches = [
                os.path.join(self.data_dir, 'substitutions_MSAs_all_positions/A4_HUMAN_theta_0.2.npy')
            ]
        if "P53" in dataset:
            weight_matches = [
                os.path.join(self.data_dir, 'substitutions_MSAs_all_positions/P53_HUMAN_theta_0.2.npy')
            ]
        assert len(weight_matches) == 1
        res_matches = [f for f in res_files if dataset == os.path.splitext(os.path.basename(f))[0]]
        assert len(res_matches) == 1

        msa_fn = msa_matches[0]
        mut_fn = mut_matches[0]
        weight_fn = weight_matches[0]
        res_fn = res_matches[0]
        mut_df = pd.read_csv(mut_fn)
        res_df = pd.read_csv(res_fn)

        y_dms = res_df.DMS_score.to_numpy()
        if dataset == 'SCN5A_HUMAN_Glazer_2019':
            mut_seqs = mut_df.mutated_sequence.map(
                lambda x: [aa_to_i[x[i]] for i in range(len(x))]).to_list()
            mut_seqs = np.asarray(mut_seqs)
            mut_seqs = mut_seqs[:, 1610:1642]
        elif dataset == 'POLG_HCVJF_Qi_2014':
            mut_seqs = mut_df.mutated_sequence.map(
                lambda x: [aa_to_i[x[i]] for i in range(len(x))]).to_list()
            mut_seqs = np.asarray(mut_seqs)
            mut_seqs = mut_seqs[:, 1983:2089]
        elif dataset == 'POLG_CXB3N_Mattenberger_2021':
            mut_seqs = mut_df.mutated_sequence.map(
                lambda x: [aa_to_i[x[i]] for i in range(len(x))]).to_list()
            mut_seqs = np.asarray(mut_seqs)
            mut_seqs = mut_seqs[:, :861]
        elif dataset == 'KCNH2_HUMAN_Kozek_2020':
            mut_seqs = mut_df.mutated_sequence.map(
                lambda x: [aa_to_i[x[i]] for i in range(len(x))]).to_list()
            mut_seqs = np.asarray(mut_seqs)
            mut_seqs = mut_seqs[:, 534:534 + 31]
        elif dataset == 'A0A140D2T1_ZIKV_Sourisseau_growth_2019':
            mut_seqs = mut_df.mutated_sequence.map(
                lambda x: [aa_to_i[x[i]] for i in range(len(x))]).to_list()
            mut_seqs = np.asarray(mut_seqs)
            mut_seqs = mut_seqs[:, 280:804]
        else:
            mut_seqs = mut_df.mutated_sequence.map(
                lambda x: [aa_to_i[x[i]] for i in range(len(x))]).to_list()

        mut_seqs = torch.tensor(mut_seqs)
        res_df = pd.read_csv(res_fn)
        weights = np.load(weight_fn)
        return msa_fn, weights, res_df, mut_seqs


def column_to_percentiles(column):
    sorted_col = torch.sort(column).values
    ranks = torch.arange(1, len(sorted_col) + 1).float()
    percentiles = 100. * ranks / len(sorted_col)
    
    # Create a dictionary to map values to their percentiles
    value_to_percentile = {val.item(): perc.item() for val, perc in zip(sorted_col, percentiles)}
    
    # Apply mapping to the original column
    percentile_col = torch.tensor([value_to_percentile[val.item()] for val in column])
    
    return percentile_col

def tensor_to_percentiles(tensor):
    n, d = tensor.shape
    percentile_tensor = torch.empty_like(tensor)

    for j in range(d):
        percentile_tensor[:, j] = column_to_percentiles(tensor[:, j])

    return percentile_tensor

def compute_weights(seqs_onehot, theta=0.2):
    zero_gaps = seqs_onehot.clone()
    zero_gaps[:, :, 0] = 0.0
    list_seq = zero_gaps.reshape(zero_gaps.shape[0], -1).cuda().to(torch.float)
    bs = 1000
    all_weights = []
    for i in range(int(np.ceil(len(list_seq) / bs))):
        batch = list_seq[i*bs:(i+1)*bs]
        length = batch.sum(dim=-1).unsqueeze(dim=-1)
        denom = torch.matmul(batch, list_seq.T) / length
        denom = (denom > 1 - theta).sum(dim=-1)
        weights = (1 / torch.clamp(denom, min=1))
        all_weights.append(weights)
    weights = torch.cat(all_weights)
    return weights

def average_empirical_variance(one_hot_matrix):
    # Calculate mean for each position/column
    mean_per_position = torch.mean(one_hot_matrix, dim=0)
    
    # Calculate variance for each position/column
    variance_per_position = torch.mean((one_hot_matrix - mean_per_position) ** 2, dim=0)
    
    # Compute the average variance across all positions
    avg_variance = torch.mean(variance_per_position)
    
    return avg_variance