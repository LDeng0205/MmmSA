import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import hessian
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from potts_mod import Potts
import scipy.stats as stats

from utils import utils

vjp = torch.autograd.functional.vjp
jvp = torch.autograd.functional.jvp

def symmetric_parametrize(J, L, A):
    """ Make J into a symmetric matrix """
    mask = (torch.ones(L, L) - torch.eye(L))[:, :, None, None]
    mask = (torch.ones(L, L, A, A) * mask).transpose(1, 2).reshape(
            L * A, L * A).to(J.device)
    J = (J.triu() + J.triu(1).transpose(-1, -2)) * mask
    # assert torch.all(J == J.T)
    return J

def pseudolikelihood(J, h, X, L, A, mask=None):
    assert torch.all(torch.isclose(J, J.T)), "Pairwise weights is not symmetric"
    tmp = (((torch.flatten(X, 1, 2) @ J) / 1) + h).reshape(
        -1, L, A)
    tmp = X * F.log_softmax(tmp, dim=2)
    if not (mask is None):
        tmp = tmp * mask.unsqueeze(-1)
    return tmp.reshape(-1, L * A).sum(dim=1)

def forward(J, h, X, L, A):
    J = J.reshape(L*A, L*A)
    assert torch.all(torch.isclose(J, J.T)), "Pairwise weights is not symmetric"
    # J = symmetric_parametrize(J, L, A)
    energy = X * ((torch.flatten(X, 1, 2) @ J / 2.0) +
                    h).reshape(-1, L, A)
    return energy.reshape(-1, L * A).sum(dim=1)

def potts_loss(J, h, w, X, lam):
    """ forward pass on the potts model to compute loss
    Args:
        J, h (array-like): Potts model parameters
        X: batched sequence data
        w: weight of each sequence
        lam (float): regularization strength, hyperparameter
    Returns:
        float: pseudolikelihood of the Potts model
    """
    L, A = X.shape[1], X.shape[2]
    J = J.reshape(L*A, L*A)

    assert torch.all(torch.isclose(J, J.T)), "Pairwise weights is not symmetric"
    mask = X.argmax(dim=-1) != 0
    mask_w = mask.sum(dim=-1) / mask.shape[-1]
    pl = pseudolikelihood(J, h, X, L, A, mask=mask) * mask_w
    loss = -(w * pl).sum()

    # regularization
    reg_loss = (h ** 2).sum() * lam
    reg_loss += 1/2 * ((J ** 2).sum()/2) * lam * (L-1) * (A-1)
    loss += reg_loss
    return loss

def fixed_point_solver_sgd_batched(train_w, train_X, orig_weights, lam=1e-7, device='cuda', batch_size=1000, 
                           n_epoch=500, lr=1e-1, verbose=False, opt='sgd', tolerance=1e-2, use_wrs=False):

    L, A = train_X.shape[1], train_X.shape[2]
    model = Potts(L, A, wt_enc=None)
    assert train_X.device == train_w.device
    assert train_X.shape[0] == train_w.shape[0]

    bs = min(len(train_X), batch_size)
    wrs = torch.utils.data.WeightedRandomSampler(orig_weights / orig_weights.sum(), orig_weights.numel(), replacement=True)
    train_dl = None
    if use_wrs:
        print('Using weighted random sampling in full potts model eval.')
        train_dl = DataLoader(TensorDataset(train_X, train_w), batch_size=bs, sampler=wrs)
    else:
        print('NOT using weighted random sampling in full potts model eval.')
        train_dl = DataLoader(TensorDataset(train_X, train_w), batch_size=bs)

    model.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=lr) if opt == 'sgd' else torch.optim.Adam(model.parameters(), lr=lr)
    
    epoch_iter = tqdm(range(n_epoch), total=n_epoch, position=0) if verbose else range(n_epoch)
    prev_loss = float('inf')

    for epoch in epoch_iter:
        epoch_l = 0.0
        model.train()
        for X, w in train_dl:
            X, w = X.to(device), w.to(device)
            def closure():
                optim.zero_grad()
                scale = bs / len(train_X)
                mask = X.argmax(dim=-1) != 0
                mask_w = mask.sum(dim=-1) / mask.shape[-1]
                pl = model.pseudolikelihood(X, mask=mask) * mask_w
                loss = -(w * pl).sum()
                # regularization
                W = model.W.weight
                reg_loss = (model.h ** 2).sum() * lam
                reg_loss += 1/2 * lam * (X.shape[1]-1) * (X.shape[2]-1) * ((W ** 2).sum()/2)
                loss += reg_loss * scale
                loss.backward()
                return loss
            # optim.step(closure)
            epoch_l += closure().item()
            optim.step()

        if epoch > 0 and abs(epoch_l - prev_loss) < tolerance:
            if verbose:
                print(f'Convergence reached after {epoch + 1} epochs.')
            break
        
        prev_loss = epoch_l
        
        if verbose:
            epoch_iter.set_postfix(epoch_loss=epoch_l, mem=torch.cuda.memory_allocated())

    model.eval()
    return -model.W.weight.detach(), -model.h.detach()
    
def fixed_point_solver_sgd(w, X, lam=1e-7, device='cuda', batch_size=1000, n_epoch=500, 
                           lr=1e-1, verbose=False, tolerance=1e-2, warmstart_params=None, opt='sgd'):
    L, A = X.shape[1], X.shape[2]
    model = None
    if warmstart_params is None:
        model = Potts(L, A, wt_enc=None)
    else:
        if verbose: print('Using warm start params.')
        model = Potts(h=warmstart_params[1].reshape(L, A), W=warmstart_params[0], wt_enc=None)

    J_opt, h_opt = train_sgd(model, X, w, lam, bs=batch_size, n_epoch=n_epoch, lr=lr, tolerance=tolerance, verbose=verbose, opt=opt, device=device)

    assert J_opt.ndim == 2

    return J_opt, h_opt

def train_sgd(model, train_X, train_w, lam, bs=1000, lr=1e-2, tolerance=1e-2, verbose=False, opt='sgd', device='cuda', n_epoch=10):
    assert train_X.device == train_w.device
    assert train_X.shape[0] == train_w.shape[0]
    # bs = min(len(train_X), bs)
    # train_dl = DataLoader(TensorDataset(train_X, train_w), batch_size=bs)
    model.to(device)
    
    optim = None
    if opt == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=lr)
    elif opt == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise NotImplementedError
    
    epoch_iter = tqdm(range(n_epoch), total=n_epoch, position=0) if verbose else range(n_epoch)
    prev_loss = float('inf') 

    X, w = train_X.to(device), train_w.to(device)
    # scale = train_X.shape[0] / 
    for epoch in epoch_iter:
        epoch_l = 0.0
        model.train()
        # for X, w in train_dl:
        optim.zero_grad()
        mask = X.argmax(dim=-1) != 0 # mask for gaps
        mask_w = mask.sum(dim=-1) / mask.shape[-1]
        with torch.enable_grad():
            pl = model.pseudolikelihood(X, mask=mask) * mask_w
            assert torch.isnan(pl).any() == False, "NaN values in pseudolikelihood calculation"
            loss = -(w * pl).sum()
            # regularization
            W = model.W.weight
            reg_loss = (model.h ** 2).sum() * lam
            reg_loss += 1/2 * ((W ** 2).sum()/2) * lam * (X.shape[1]-1) * (X.shape[2]-1)
            loss += reg_loss 
            loss.backward()
            optim.step()
        epoch_l += loss.item()
        assert torch.isnan(loss).any() == False, "Potts model loss is NaN"
        
        if epoch > 0 and abs(epoch_l - prev_loss) < tolerance:
            if verbose:
                print(f'Convergence reached after {epoch + 1} epochs.')
            break
        
        prev_loss = epoch_l
        
        if verbose:
            epoch_iter.set_postfix(epoch_loss=epoch_l, mem=torch.cuda.memory_allocated())

    model.eval()
    return -model.W.weight.detach(), -model.h.detach()

def jvp_dwdJ_loss(J, h, X, w, L, A, v):
    mask = X.argmax(dim=-1) != 0
    mask_w = mask.sum(dim=-1) / mask.shape[-1]
    # pl_J = lambda J : mask_w * pseudolikelihood(symmetric_parametrize(J.reshape(L*A, L*A), L, A), h, X, L, A, mask=mask)
    pl_J = lambda J : mask_w * pseudolikelihood(J.reshape(L*A, L*A), h, X, L, A, mask=mask)
    comp_dJ_loss = lambda w: -vjp(pl_J, J, w, create_graph=True)[1]/w.sum()
    return vjp(comp_dJ_loss, w, v)[1]

def jvp_dwdh_loss(J, h, X, w, L, A, v):
    mask = X.argmax(dim=-1) != 0
    mask_w = mask.sum(dim=-1) / mask.shape[-1]
    # pl_h = lambda h : mask_w * pseudolikelihood(symmetric_parametrize(J.reshape(L*A, L*A), L, A), h, X, L, A, mask=mask)
    pl_h = lambda h : mask_w * pseudolikelihood(J.reshape(L*A, L*A), h, X, L, A, mask=mask)
    comp_dh_loss = lambda w: -vjp(pl_h, h, w, create_graph=True)[1]/w.sum()
    return vjp(comp_dh_loss, w, v)[1]

def pearson_r(J, h, dms_X, dms_y, L, A):

    assert torch.all(torch.isclose(J, J.T)), "Pairwise weights is not symmetric"
    x = forward(J.reshape(L*A, L*A), h, dms_X, L, A)
    vx = x - torch.mean(x)
    vy = dms_y - torch.mean(dms_y)

    r = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return r

def spearman_r(J, h, dms_X, dms_y, L, A):
    assert torch.all(torch.isclose(J, J.T)), "Pairwise weights is not symmetric"
    x = forward(J.reshape(L*A, L*A), h, dms_X, L, A)
    r = utils.diff_spearmanr(x.unsqueeze(0).cpu(), dms_y.unsqueeze(0).cpu())
    return r

def spearman_r_no_grad(J, h, dms_X, dms_y, L, A):
    assert torch.all(torch.isclose(J, J.T)), "Pairwise weights is not symmetric"
    x = forward(J.reshape(L*A, L*A), h, dms_X, L, A)
    r, p_value = stats.spearmanr(x.cpu().detach().numpy(), dms_y.cpu().detach().numpy())
    return r

class ImplicitPotts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, X, sketch_dim, args):
        w = w.detach().clone()

        J_opt, h_opt = fixed_point_solver_sgd(w, X, **args)
        assert torch.all(torch.isclose(J_opt, J_opt.T)), "Pairwise weights is not symmetric"
        ctx.save_for_backward(J_opt, h_opt, w, X)
        ctx.lam = args['lam']
        ctx.device = args['device']
        ctx.sketch_dim = sketch_dim
        
        return J_opt, h_opt
    
    @staticmethod
    def backward(ctx, grad_J, grad_h):
        
        device = ctx.device
        lam = ctx.lam
        J_opt, h_opt, train_w, train_X = ctx.saved_tensors
        train_X = train_X.to(ctx.device)
        
        L, A = train_X.shape[1], train_X.shape[2]
        
        with torch.no_grad():
            # Hessian of loss, where we use matrix sketching to project the Hessian on to a random subspace of R^n
            # with sketch dimension m; i.e. higher m -> exact solution
            f_s1 = lambda l, S, w : potts_loss(J_opt.flatten() + S @ l, h_opt, w, train_X, lam)
            f_s2 = lambda l, S, w : potts_loss(J_opt.flatten(), h_opt + S @ l, w, train_X, lam)

            m = ctx.sketch_dim
            fail_cnt = 0
            d_train_w = torch.zeros_like(train_w)
            for _ in range(2):
                # print(f'Computing approximation #{i+1} for gradient')
                while fail_cnt < 10:
                    try: 
                        # Gaussian Sketch
                        S1 = torch.randn(L * A * L * A, min(m, L*A*L*A), device=device) / torch.sqrt(torch.tensor(m, device=device))
                        S2 = torch.randn(L * A, min(m, L*A), device=device) / torch.sqrt(torch.tensor(m, device=device))
                        
                        l1 = torch.zeros(min(m, L*A*L*A), device=device) # need to get hessian at l1 = 0 to approximate the un-sketched hessian
                        l2 = torch.zeros(min(m, L*A), device=device) # need to get hessian at l2 = 0 to approximate the un-sketched hessian

                        H_s1 = hessian(f_s1, argnums=(0))(l1, S1, train_w) # S1.T @ H_W @ S1 where H_W is the hessian of potts loss w.r.t J
                        H_s2 = hessian(f_s2, argnums=(0))(l2, S2, train_w) # S2.T @ H_h @ S2 where H_h is the hessian of potts loss w.r.t h

                        # H_s1 and H_s2 should be symmetric, since Hessian of a convex function is PSD
                        # assert torch.allclose(H_s1, H_s1.T)
                        # assert torch.allclose(H_s2, H_s2.T)

                        # solve the linear system to get gradient of train_w
                        x_1 = torch.linalg.solve(H_s1, S1.T @ grad_J.flatten())
                        x_2 = torch.linalg.solve(H_s2, S2.T @ grad_h.flatten())

                        # multiply by sketch matrix to get adjoint variables
                        u_1 = S1 @ x_1
                        u_2 = S2 @ x_2

                        # multiply adjoint variables by jacobian of fixed function w.r.t. W and h
                        grad1 = -jvp_dwdJ_loss(J_opt.flatten(), h_opt, train_X, train_w, L, A, u_1)
                        grad2 = -jvp_dwdh_loss(J_opt.flatten(), h_opt, train_X, train_w, L, A, u_2)

                        d_train_w_ = grad1 + grad2
                        break
                    except Exception as e:
                        fail_cnt += 1
                        if 'out of memory' in str(e):
                            m = int(m*0.7)
                            print(f'| WARNING: ran out of memory, reducing sketch dim to {m}')
                        else:
                            raise e
                d_train_w += d_train_w_
        d_train_w /= 2
        return d_train_w, None, None, None
    
class ImplicitPottsNoSketch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, X, sketch_dim, args):
        w = w.detach().clone()

        J_opt, h_opt = fixed_point_solver_sgd(w, X, **args)
        
        ctx.save_for_backward(J_opt, h_opt, w, X)
        ctx.lam = args['lam']
        ctx.device = args['device']
        ctx.sketch_dim = sketch_dim
        
        return J_opt, h_opt
    
    @staticmethod
    def backward(ctx, grad_J, grad_h):
        
        device = ctx.device
        lam = ctx.lam
        J_opt, h_opt, train_w, train_X = ctx.saved_tensors
        train_X = train_X.to(ctx.device)
        
        L, A = train_X.shape[1], train_X.shape[2]

        device = ctx.device
        
        with torch.no_grad():
            L_W = lambda W, train_w : potts_loss(W, h_opt, train_w, train_X, lam)
            L_h = lambda h, train_w : potts_loss(J_opt, h, train_w, train_X, lam)

            dLdW = hessian(L_W, argnums=(0, 1))(J_opt.flatten(), train_w)
            dLdh = hessian(L_h, argnums=(0, 1))(h_opt, train_w)

            print(torch.sum(dLdW[0][0]))
            
            grad1 = grad_J.flatten() @ torch.linalg.inv(dLdW[0][0]) @ dLdW[0][1]
            grad2 = grad_h.flatten() @ torch.linalg.inv(dLdh[0][0]) @ dLdh[0][1]
            d_train_w = grad1 + grad2
    
        return d_train_w, None, None, None