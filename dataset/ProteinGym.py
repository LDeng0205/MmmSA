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


class ProteinGym(Dataset):
    def __init__(self, name, alphabet=None, balance=False, features=None, domain_only=False,
                 orig_weights_cutoffs=None, normalization='standard', pc_path=None, random_permute=False,
                 esm_embeddings_path='/home/hanlunj/projects/protein_gym/20231011_esm_embeddings', #TODO: implement
                 data_dir='/home/arthur/data/ProteinGym2024'):
        if alphabet is None:
            self.alphabet = '-ACDEFGHIKLMNPQRSTVWY'
        else:
            self.alphabet = alphabet

        self.data_dir = data_dir

        df = pd.read_csv(os.path.join(data_dir, "DMS_substitutions.csv"))
        msa_name, dms_name, weight_name = df.loc[df['DMS_id'] == name, ['MSA_filename', 'DMS_filename', 'weight_file_name']].to_numpy()[0]
        msa_fn, weights, y_dms, mut_seqs = self.get_proteingym_data(msa_name, dms_name, weight_name)
        
        # Process MSA sequences
        msa_sequences_raw = [str(x.seq) for x in SeqIO.parse(msa_fn, 'fasta')]
        msa_sequences_raw = [s.replace(".", "-").upper() for s in msa_sequences_raw]
        wt_seq = msa_sequences_raw[0]
        self.wt_raw = wt_seq
        
        columns_to_keep = [i for i in range(len(wt_seq))]

        msa_sequences_raw = [s for s in msa_sequences_raw if self.check_sequence(s)]

        msa_sequences = [[s[i] for i in columns_to_keep] for s in msa_sequences_raw]
        msa_sequences = np.asarray(msa_sequences)

        seqs_enc, aa_to_i = self.encode(msa_sequences)

        i_to_a = {i:aa for i, aa in enumerate(self.alphabet)}

        assert weights.shape[0] == len(msa_sequences)

        seqs = torch.nn.functional.one_hot(seqs_enc, num_classes=len(self.alphabet)).to(torch.float32)
        
        # Process DMS sequences
        mut_seqs_onehot = torch.nn.functional.one_hot(mut_seqs, num_classes=len(self.alphabet)).to(torch.float32)

        self.seqs = seqs
        
        self.name = msa_name
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
                if feature == 'gap':
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
                    feature[idx] = vars.unsqueeze(1)
                    self.data = torch.hstack((self.data, feature.unsqueeze(-1)))
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
        
    def get_proteingym_data(self, msa_name, dms_name, weight_name):
        aa_to_i = {aa: i for i, aa in enumerate(self.alphabet)}

        msa_file = os.path.join(self.data_dir, 'MSA', msa_name)
        mutation_file = os.path.join(self.data_dir, 'DMS', dms_name)
        weight_file = os.path.join(self.data_dir, 'MSA_weights', weight_name)

        assert os.path.exists(msa_file)
        assert os.path.exists(mutation_file)
        assert os.path.exists(weight_file)

        mut_df = pd.read_csv(mutation_file)
        y_dms = mut_df.DMS_score.to_numpy()
        mut_seqs = mut_df.mutated_sequence.map(lambda x: [aa_to_i[x[i]] for i in range(len(x))]).to_list()

        mut_seqs = torch.tensor(mut_seqs)
        weights = np.load(weight_file)
        return msa_file, weights, y_dms, mut_seqs


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