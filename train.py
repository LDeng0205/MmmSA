import os
import torch
import numpy as np
import pandas as pd
import gc
import wandb
import copy

from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, ExponentialLR

from model.model import *
from implicit_potts import *
from utils.utils import *
from dataset.ProteinGym import *

import pickle

from argparse import ArgumentParser
from datetime import datetime

def train(model, dataset_list, potts_args, loss_class='spearman', opt='sgd', lr=1e-2, bs=1000, sketch_dim=20, 
          n_epoch=20, log_wandb=False, weighted_sampling=False, no_shuffle=False, 
          num_batches=1, warmstart=False, val_name_list=None, args=None, device='cuda'):

    optim = None
    if opt == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=lr)
    elif opt == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    scheduler = None
    if args.scheduler == 'exp':
        scheduler = ExponentialLR(optim, gamma=0.90)
    elif args.scheduler == 'step':
        scheduler = StepLR(optim, step_size=30, gamma=0.1)

    epochs = tqdm(range(n_epoch))
    losses = []

    if warmstart:
        warmstart_params = {dataset.name: None for dataset in dataset_list}

    
    for epoch in epochs:
        logs = {}
        epoch_l = 0.0
        optim.zero_grad()

        for ds in dataset_list:
            if args.load_ds_on_demand:
                dataset = ProteinGym(ds, features=args.features, domain_only=args.domain_only, orig_weights_cutoffs=args.theta, 
                                normalization=args.normalization, pc_path=args.pc_path, esm_embeddings_path=args.esm_embeddings_path, 
                                data_dir=args.data_dir, random_permute=args.random_permute)
            else:
                dataset = ds
            print(f'Training with {dataset.name}')
            model.train()
            if weighted_sampling:
                if no_shuffle:
                    # rng_state = torch.get_rng_state()
                    torch.manual_seed(args.seed)
                loader = DataLoader(dataset, batch_size=min(bs, len(dataset)), drop_last=True, 
                                  sampler=torch.utils.data.WeightedRandomSampler(dataset.orig_weights / dataset.orig_weights.sum(), 
                                                                                 dataset.orig_weights.numel(), replacement=True))
                # if no_shuffle:
                #     torch.set_rng_state(rng_state)

            else:
                if no_shuffle:
                    torch.manual_seed(args.seed)
                loader = DataLoader(dataset, batch_size=min(bs, len(dataset)), shuffle=(not no_shuffle), drop_last=False)
                
            X_dms, y_dms = dataset.get_dms()
            X_dms, y_dms = X_dms.to(device), y_dms.to(device)
            L, A = dataset.get_LA()

            dataset_name, dataset_epoch_l, batch_cnt = dataset.name, 0.0, 0
            weight_diff = 0 # distance between original weights and predicted weights
            for x_msa, x_d, x_orig_w in loader:
                try:
                    x_d, x_msa, x_orig_w = x_d.to(device), x_msa.to(device), x_orig_w.to(device)
                    with torch.enable_grad():
                        # Obtain predicted weights from the metamodel
                        weights = None
                        # print(x_orig_w.shape)
                        assert torch.isnan(x_orig_w).any() == False, "orig weights contains NaN"
                        assert torch.isnan(x_d).any() == False, 'features contains NaN'

                        if (args.model_class == 'nn_modulate' or args.model_class == 'big_nn_modulate' 
                            or args.model_class == 'log_reg_modulate' or args.model_class == 'lin_reg_modulate'):
                            weights = model(x_d, x_orig_w).squeeze()
                        else:
                            weights = model(x_d).squeeze()
                        # print(weights.shape)
                        assert weights.ndim==1            
                        assert torch.isnan(weights).any() == False, "predicted weights contains NaN"

                        # Train a potts model (implicit layer)
                        dataset_potts_args = copy.deepcopy(potts_args)
                        dataset_potts_args['lam'] *= bs/len(dataset) # scale
                        
                        if warmstart:
                            dataset_potts_args['warmstart_params'] = warmstart_params[dataset.name]

                        J_opt, h_opt = ImplicitPotts.apply(weights, x_msa, sketch_dim, dataset_potts_args)

                        if warmstart:
                            warmstart_params[dataset.name] = (J_opt.clone().detach().cpu(), 
                                                                                  h_opt.clone().detach().cpu())
                        loss = None
                        # Compute the loss on DMS data
                        if loss_class == 'pearson':
                            loss = -pearson_r(J_opt.reshape(L*A, L*A), h_opt, X_dms, y_dms, L, A)
                        elif loss_class == 'spearman':
                            loss = -spearman_r(J_opt.reshape(L*A, L*A), h_opt, X_dms, y_dms, L, A)
                        print(f'{loss_class} loss for batch: ', loss.item())

                        epoch_l += loss
                        weight_diff += torch.dist(weights, x_orig_w).item()

                    with torch.no_grad():
                        dataset_epoch_l += loss.item()
                        
                except RuntimeError as e:
                    print(str(e))
                    if 'out of memory' in str(e):
                        print('| WARNING: ran out of memory, skipping batch')
                        for p in model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        raise e
                batch_cnt += 1
                
                if batch_cnt >= num_batches:
                    break
            
            if args.train_val_freq != -1 and ((epoch+1) % args.train_val_freq == 0 or args.run_baseline):
                # Evaluate by training a potts model with the entire dataset
                print('Training a potts model with the metamodel and a full MSA')
                msa_seqs, featurized_seqs = dataset.seqs.to(device), dataset.data.to(device)
                model.eval()
                weights = None
                if args.run_baseline:
                    print('Using original weights for eval')
                    weights = dataset.orig_weights.to(device)
                    if args.val_freq == -1:
                        args.run_baseline = False
                else:
                    if (args.model_class == 'nn_modulate' or args.model_class == 'big_nn_modulate' 
                            or args.model_class == 'log_reg_modulate' or args.model_class == 'lin_reg_modulate'):
                        weights = model(featurized_seqs, dataset.orig_weights.to(device)).squeeze()
                    else:
                        weights = model(featurized_seqs).squeeze()
                    weights = weights.detach()

                J_opt, h_opt = fixed_point_solver_sgd_batched(weights, msa_seqs, dataset.orig_weights, lam=1e-2, device=device, batch_size=1000, 
                            n_epoch=500, lr=3e-4, verbose=True, opt='sgd', tolerance=1e-3, use_wrs=args.use_wrs_full_potts) # the numbers Hanlun used for scanning blosum cutoff for potts spearman
                
                spearman = spearman_r_no_grad(J_opt, h_opt, X_dms, y_dms, L, A)
                pearson = pearson_r(J_opt, h_opt, X_dms, y_dms, L, A)
                if log_wandb:
                    logs.update({f'{dataset_name}_full_MSA_spearman_estimate': spearman}, step=epoch+1)
                    logs.update({f'{dataset_name}_full_MSA_pearson_estimate': pearson}, step=epoch+1)

            if log_wandb:
                logs.update({f'{dataset_name}_{loss_class}_loss': dataset_epoch_l}, step=epoch+1)
                logs.update({f'{dataset_name}_weight_diff': weight_diff}, step=epoch+1)

        epoch_l.backward()
        optim.step()

        if args.model_save_freq != -1 and (epoch + 1) % args.model_save_freq == 0:
            print('Saving model')
            for name, param in model.named_parameters():
                print(f"Parameter name: {name}")
                print(f"Parameter value: {param}")
            save_path = os.path.join(args.workdir, args.run_name, f'metamodel_epoch{epoch+1}.pt') # directory to save model
            torch.save(model.state_dict(), save_path)
        if not args.scheduler is None:
            scheduler.step()
        
        if args.val_freq != -1 and ((epoch+1) % args.val_freq == 0 or args.run_baseline or epoch == 1):
            model.eval()
            val_metrics = test_epoch(val_name_list, args)
            logs.update(val_metrics, step=epoch+1)

        with torch.no_grad():
            losses.append(epoch_l.item())
            epochs.set_postfix(mem=torch.cuda.memory_allocated() , epoch_l=epoch_l.item())
            
            if log_wandb:
                logs.update({f'train_epoch_{loss_class}_loss': epoch_l}, step=epoch+1)
                if not args.scheduler is None:
                    current_lr = scheduler.get_last_lr()[0]
                    logs.update({f'learning rate': current_lr}, step=epoch+1)
                wandb.log(logs, step=epoch + 1)

        print(f'Epoch {epoch + 1} complete')

    model.eval()
    return model

def test_epoch(names, args):
    metrics = {}
    average_sp = 0
    average_p = 0
    for name in names:
        print(f'Running full potts model evaluation on dataset {name}')
        dataset = ProteinGym(name, features=args.features, domain_only=args.domain_only, orig_weights_cutoffs=args.theta, 
                                    normalization=args.normalization, pc_path=args.pc_path, esm_embeddings_path=args.esm_embeddings_path, 
                                    data_dir=args.data_dir, random_permute=args.random_permute)
        
        msa_seqs, featurized_seqs = dataset.seqs.to(device), dataset.data.to(device)
        model.eval()
        weights = None
        if args.run_baseline:
            print('Using original weights for eval')
            weights = dataset.orig_weights.to(device)
        else:
            if (args.model_class == 'nn_modulate' or args.model_class == 'big_nn_modulate' or 
                args.model_class == 'log_reg_modulate' or args.model_class == 'lin_reg_modulate'):
                weights = model(featurized_seqs, dataset.orig_weights.to(device)).squeeze()
            else:
                weights = model(featurized_seqs).squeeze()
            weights = weights.detach()
        if not args.eval_seed is None:
            torch.manual_seed(args.eval_seed)
        J_opt, h_opt = fixed_point_solver_sgd_batched(weights, msa_seqs, dataset.orig_weights, lam=1e-2, device=device, batch_size=1000, 
                    n_epoch=500, lr=3e-4, verbose=True, opt='sgd', tolerance=1e-3, use_wrs=args.use_wrs_full_potts) # the numbers Hanlun used for scanning blosum cutoff for potts spearman
        
        X_dms, y_dms = dataset.get_dms()
        X_dms, y_dms = X_dms.to(device), y_dms.to(device)
        L, A = dataset.get_LA()

        spearman = spearman_r_no_grad(J_opt, h_opt, X_dms, y_dms, L, A)
        pearson = pearson_r(J_opt, h_opt, X_dms, y_dms, L, A)
        
        metrics[f'{name}_full_MSA_spearman_estimate'] = spearman
        metrics[f'{name}_full_MSA_pearson_estimate'] = pearson
        average_p += pearson
        average_sp += spearman

    if args.run_baseline:
        args.run_baseline = False
        
    metrics['Average Test Spearman'] = average_sp/len(names)
    metrics['Average Test Pearson'] = average_p/len(names)
        
    return metrics

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--pretrain_path', type=str, default=None, help='Path to load existing model from')
    parser.add_argument('--workdir', type=str, default='workdir', help='Path to folder to save the meta model')
    parser.add_argument('--run_name', type=str, default='Metamodel_train', help='Wandb run name')
    parser.add_argument('--train_val_freq', type=int, default=5, help='')
    parser.add_argument('--val_freq', type=int, default=-1, help='')
    parser.add_argument('--seed', type=int, default=17, help='')
    parser.add_argument('--data_dir', type=str, default='/home/arthur/data/ProteinGym2024', help='')
    parser.add_argument('--model_save_freq', type=int, default=10, help='')
    parser.add_argument('--eval_seed', type=int, default=None, help='')

    # Dataset
    parser.add_argument('--dataset', type=str, default=None, help='')
    parser.add_argument('--split_train', type=str, default='data/splits/train.txt', 
                        help='Path to .txt file containing names of Protein Gym datasets used for training')
    parser.add_argument('--split_test', type=str, default='data/splits/val.txt', 
                        help='Path to .txt file containing names of Protein Gym datasets used for validation')
    parser.add_argument('--pc_path', type=str, default=None, 
                        help='Path to .pt file containing precomputed principal compoenents for esm embeddings')
    parser.add_argument('--subsample', action='store_true', default=False, help='Use subsampled MSA to train')
    parser.add_argument('--subsample_size', type=int, default=3000, help='Size of subsampled dataset') #TODO: remove
    parser.add_argument('--balance', action='store_true', default=False, help='Use balanced sampling') #TODO: remove
    parser.add_argument('--balance_cutoff', type=float, default=0.5, help='') #TODO: remove
    parser.add_argument('--balance_feature', type=str, default=None, help='Available features are: hamming,hamming_square,blosum,gap,functional_sites') #TODO: remove
    parser.add_argument('--weighted_sampling', action='store_true', default=False, help='')
    parser.add_argument('--theta', type=float, nargs='+', default=[0.2], help='')
    parser.add_argument('--normalization', type=str, default='standard', help='')
    parser.add_argument('--features', type=str, nargs='+', default='', help='Features to use; overwrites --num_features. '+\
                                                    'Available features are: hamming,hamming_square,blosum,gap,functional_sites')
    parser.add_argument('--domain_only', action='store_true', default=False, help='')
    parser.add_argument('--esm_embeddings_path', type=str, default='/home/hanlunj/projects/protein_gym/20231011_esm_embeddings', help='')
    parser.add_argument('--random_permute', action='store_true', default=False, help='')
    parser.add_argument('--load_ds_on_demand', action='store_true', default=False, help='')

    # MetaModel
    parser.add_argument('--model_class', type=str, default='log_reg', help='Model type')
    parser.add_argument('--num_features', type=int, default=2, help='')

    # MetaModel training
    parser.add_argument('--optimizer', type=str, default='sgd', help='Meta model optimizer')
    parser.add_argument('--lr', type=float, default=1e-1, help='Meta model lr')
    parser.add_argument('--n_epoch', type=int, default=20, help='Meta model train epochs')
    parser.add_argument('--batch_size', type=int, default=3000, help='')
    parser.add_argument('--no_shuffle', action='store_true', default=False, help='Do not shuffle data for every epoch')
    parser.add_argument('--num_batches', type=int, default=1, help='Max number of batches to use per epoch')
    parser.add_argument('--loss', type=str, default='spearman', help='')
    parser.add_argument('--warmstart', action='store_true', default=False, help='')
    parser.add_argument('--scheduler', type=str, default=None, help='Learning rate scheduler')

    # Potts params
    parser.add_argument('--sketch_dim', type=int, default=20, help='Sketch dimension used in the backward pass of the implicit layer')
    parser.add_argument('--potts_n_epoch', type=int, default=1000, help='Potts model max train epochs')
    parser.add_argument('--potts_batch_size', type=int, default=1000, help='Potts model batch size')
    parser.add_argument('--potts_tol', type=float, default=1e-2, help='Convergence criterion for potts model')
    parser.add_argument('--potts_lr', type=float, default=1e-1, help='Potts model lr')
    parser.add_argument('--potts_lam', type=float, default=1e-2, help='Potts model regularization strength')
    parser.add_argument('--potts_opt', type=str, default='sgd', help='')
    
    parser.add_argument('--wandb', action='store_true', default=False, help='Use wandb to log training')
    parser.add_argument('--cpu', action='store_true', default=False, help='Set device to cpu')

    # Validation
    parser.add_argument('--use_wrs_full_potts', action='store_true', default=False, help='')
    parser.add_argument('--run_baseline', action='store_true', default=False, help='')

    args = parser.parse_args()
    cache_name = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f')

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(device)
    # Set the seed
    torch.manual_seed(args.seed)

    dataset_list = []
    dataset_name_list = []

    directory_path = os.path.join(args.workdir, args.run_name) # directory to save model
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    if not args.features is None: 
        args.num_features = len(args.features)
        if 'orig_weights' in args.features:
            args.num_features -= 1
            args.num_features += len(args.theta)
        if 'esm_pca' in args.features:
            args.num_features -= 1
            args.num_features += 7
        print('Using features: ', args.features)
        print('Updating args.num_features to: ', args.num_features) 

    if args.wandb:
        wandb.init(
            entity='msa-weights2',
            project="msa-weights",
            name=args.run_name,
            # track hyperparameters and run metadata
            config={
                "epochs": args.n_epoch,
                "args": args,
            }
        )

    name_list = []

    if args.dataset is not None:
        name_list = [args.dataset]
    else:
        split_file_path = args.split_train
        with open(split_file_path, 'r') as file:
            for line in file:
                name_list.append(line.strip())

    if args.load_ds_on_demand:
        dataset_list = name_list
    else:
        dataset_list = [ProteinGym(name, features=args.features, domain_only=args.domain_only, orig_weights_cutoffs=args.theta, 
                                normalization=args.normalization, pc_path=args.pc_path, esm_embeddings_path=args.esm_embeddings_path, 
                                data_dir=args.data_dir, random_permute=args.random_permute) for name in name_list]
        
    val_name_list = None
    if args.val_freq != -1:
        val_name_list = []
        with open(args.split_test, 'r') as file:
            for line in file:
                val_name_list.append(line.strip())
        print(f'Validation datasets: {val_name_list}')

    model = None
    if args.model_class == 'log_reg':
        model = LogisticRegression(d=args.num_features)
        model = model.to(device)
    elif args.model_class == 'log_reg_modulate':
        model = LogisticRegressionModulate(d=args.num_features)
        model = model.to(device)
    elif args.model_class == 'lin_reg_modulate':
        model = LinearRegressionModulate(d=args.num_features)
        model = model.to(device)
    elif args.model_class == 'nn':
        model = TwoLayerNN(d=args.num_features)
        model = model.to(device)
    elif args.model_class == 'nn_modulate':
        model = TwoLayerNNModulate(d=args.num_features)
        model = model.to(device)
    elif args.model_class == 'big_nn_modulate':
        model = BigTwoLayerNNModulate(d=args.num_features)
        model = model.to(device)
    elif args.model_class == 'relu':
        model = LinearRegression(d=args.num_features)
        model = model.to(device)
    else:
        print('Model class not implemented!')

    if not args.pretrain_path is None:
        saved_state_dict = torch.load(args.pretrain_path)
        model.load_state_dict(saved_state_dict)
    
    potts_args = {"lam":args.potts_lam, "device":device, "batch_size":args.potts_batch_size,
                  "n_epoch":args.potts_n_epoch, "lr":args.potts_lr, "verbose":True, 
                  "tolerance":args.potts_tol, "opt":args.potts_opt}
            
    model = train(model, dataset_list, potts_args, loss_class=args.loss, lr=args.lr, sketch_dim=args.sketch_dim, n_epoch=args.n_epoch, 
                          bs=args.batch_size, opt=args.optimizer, log_wandb=args.wandb, 
                          weighted_sampling=args.weighted_sampling, no_shuffle=args.no_shuffle, warmstart=args.warmstart,
                          num_batches=args.num_batches, val_name_list=val_name_list, args=args, device=device)

    print("Training Complete!")
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Parameter value: {param}\n")

    save_path = f'{directory_path}/metamodel_final.pt'
    torch.save(model.state_dict(), save_path)
