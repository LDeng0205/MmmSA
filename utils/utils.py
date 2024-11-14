import numpy as np
import pandas as pd
import torch
# from evcouplings.align import ALPHABET_PROTEIN
from collections import OrderedDict
from torch.utils.data import TensorDataset
from Bio import SeqIO
from tqdm import tqdm

from pathlib import Path
from fast_soft_sort.pytorch_ops import soft_rank

from potts_mod import Potts

def diff_spearmanr(pred, target, **kw):
    ''' Differentiable spearman correlation using differentiable ranking algorithm. 
        see https://pypi.org/project/torchsort/. '''
    pred = soft_rank(pred, regularization_strength=0.01)
    target = soft_rank(target, regularization_strength=0.01)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()

ALPHABET_PROTEIN = '-ACDEFGHIKLMNPQRSTVWY'

# Loading datasets
def check_sequence(s, alphabet=ALPHABET_PROTEIN):
    for aa in s:
        if aa not in ALPHABET_PROTEIN:
            return False
    return True

def encode(seqs, alphabet=ALPHABET_PROTEIN, verbose=True):
    '''
    Go from letters to numbers
    '''
    aa_to_i = OrderedDict((aa, i) for i, aa in enumerate( alphabet ))
    if verbose:
        seq_iter = tqdm(seqs)
    else:
        seq_iter = seqs
    X = torch.tensor([[aa_to_i[x] for x in seq] 
                      for seq in seq_iter])
    return X, aa_to_i