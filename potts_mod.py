# @author Hunter Nisonoff
# @author Arthur Deng

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torch.nn.functional as F


class Symmetric(nn.Module):

    def __init__(self, L, A):
        super().__init__()
        # mask out diagonal
        mask = (torch.ones(L, L) - torch.eye(L))[:, :, None, None]
        mask = (torch.ones(L, L, A, A) * mask).transpose(1, 2).reshape(
            L * A, L * A)
        self.register_buffer('mask', mask)
        mask_2 = torch.ones(L*A, L*A)
        for i in range(L):
            mask_2[i * A: (i + 1) * A, i * A: (i + 1) * A] = torch.zeros(A, A)
        self.register_buffer('mask_2', mask_2)

    def forward(self, X):
        return (X.triu() + X.triu(1).transpose(-1, -2)) * self.mask * self.mask_2

class WeightMask(nn.Module):

    def __init__(self, W_mask):
        super().__init__()
        self.register_buffer('W_mask', W_mask)

    def forward(self, X):
        return X * self.W_mask


class WTGauge(nn.Module):

    def __init__(self, W_mask):
        super().__init__()
        self.register_buffer('W_mask', W_mask)

    def forward(self, X):
        return X * self.W_mask


class Potts(torch.nn.Module):

    def __init__(self,
                 L=None,
                 A=None,
                 h=None,
                 W=None,
                 temp=1.0,
                 wt_enc=None,
                 weight_mask=None):
        super().__init__()
        if L is not None:
            assert (A is not None) and (h is None) and (W is None)
        if A is not None:
            assert (L is not None) and (h is None) and (W is None)
        if h is not None:
            assert (W is not None) and (L is None) and (A is None)
        if W is not None:
            assert (W is not None) and (L is None) and (A is None)
        # Scenario 1
        if (L is not None) and (A is not None):
            self.L = L
            self.A = A
            self.h = nn.Parameter(torch.zeros(L * A))
            self.W = nn.Linear(L * A, L * A, bias=False)
            self.W.weight.data.fill_(0.0) #TODO: should this be initialized to 0?
            self.sym = Symmetric(L, A)
            parametrize.register_parametrization(self.W, "weight", self.sym)
        else:
            h = h.clone().detach().to(torch.float)
            #h = torch.tensor(h, dtype=torch.float)
            assert (h.ndim == 2)
            self.L, self.A = h.shape
            self.h = nn.Parameter(h.reshape(-1))
            
            W = W.clone().detach().to(torch.float)
            
            assert (W.ndim == 2)
            assert W.shape[0] == W.shape[1] == self.L * self.A
            self.W = nn.Linear(self.L * self.A, self.L * self.A, bias=False)
            self.W.weight = nn.Parameter(W)

            self.sym = Symmetric(self.L, self.A)
            parametrize.register_parametrization(self.W, "weight", self.sym)

        self.temp = temp
        self.wt_enc = wt_enc
        if not (wt_enc is None):
            h_mask = torch.ones((self.L, self.A))
            for pos in range(self.L):
                wt_aa_idx = wt_enc[pos]
                h_mask[pos, wt_aa_idx] = 0
            h_mask = h_mask.reshape(-1)
            self.register_buffer('h_mask', h_mask)

            W_mask = torch.ones((self.L, self.L, self.A, self.A))
            for pos_i in range(self.L):
                for pos_j in range(self.L):
                    if pos_i == pos_j:
                        W_mask[pos_i, pos_j] = 0.0
                    else:
                        wt_aa_i = wt_enc[pos_i]
                        wt_aa_j = wt_enc[pos_j]
                        W_mask[pos_i, pos_j, wt_aa_i] = 0.0
                        W_mask[pos_i, pos_j, :, wt_aa_j] = 0.0
                        W_mask[pos_j, pos_i, wt_aa_j] = 0.0
                        W_mask[pos_j, pos_i, :, wt_aa_i] = 0.0
            W_mask = W_mask.transpose(1, 2).reshape(self.L * self.A,
                                                    self.L * self.A)

            self.wt_gauge = WTGauge(W_mask)
            parametrize.register_parametrization(self.W, "weight",
                                                 self.wt_gauge)

        if not (weight_mask is None):
            self.weight_mask = WeightMask(weight_mask)
            parametrize.register_parametrization(self.W, "weight",
                                                 self.weight_mask)

    def load_from_weights(self, h, W):
        h = h.clone().detach().to(torch.float)
        assert (h.ndim == 2)
        self.L, self.A = h.shape
        self.h = nn.Parameter(h.reshape(-1))
        W = W.clone().detach().to(torch.float)
        assert (W.ndim == 2)
        assert W.shape[0] == W.shape[1] == self.L * self.A
        self.W = nn.Linear(self.L * self.A, self.L * self.A, bias=False)
        self.W.weight = nn.Parameter(W)
        self.sym = Symmetric(self.L, self.A)
        parametrize.register_parametrization(self.W, "weight", self.sym)
        return

    def reshape_to_L_L_A_A(self):
        return self.W.weight.reshape(
            (self.L, self.A, self.L, self.A)).transpose(1, 2)

    def pseudolikelihood(self, X, mask=None):
        if not (self.wt_enc is None):
            tmp = (-((self.W(torch.flatten(X, 1, 2)) / 1) +
                     (self.h * self.h_mask))).reshape(-1, self.L, self.A)
        else:
            tmp = (-((self.W(torch.flatten(X, 1, 2)) / 1) + self.h)).reshape(
                -1, self.L, self.A)
        tmp = X * F.log_softmax(tmp, dim=2)
        if not (mask is None):
            tmp = tmp * mask.unsqueeze(-1)
        return tmp.reshape(-1, self.L * self.A).sum(dim=1)

    def marginals(self, X, mask=None):
        tmp = (-((self.W(torch.flatten(X, 1, 2)) / 2) + self.h)).reshape(
            -1, self.L, self.A)
        tmp = F.log_softmax(tmp, dim=2)
        return tmp

    def pseudolikelihood_debug(self, X, mask=None):
        tmp = (-((self.W(torch.flatten(X, 1, 2)) / 1) + self.h)).reshape(
            -1, self.L, self.A)
        return F.log_softmax(tmp, dim=2)

    def forward(self, X, beta=1.0):
        energy = X * ((self.W(torch.flatten(X, 1, 2)) * beta / 2.0) +
                      self.h).reshape(-1, self.L, self.A)
        return energy.reshape(-1, self.L * self.A).sum(dim=1)

    def single_pairwise_energy(self, X, beta=1.0):
        pairwise = (X * self.W(torch.flatten(X, 1, 2)).reshape(
            -1, self.L, self.A)).reshape(
                -1, self.L * self.A).sum(dim=1) * beta / 2.0
        single = (X * self.h.reshape(-1, self.L, self.A)).reshape(
            -1, self.L * self.A).sum(dim=1)
        return single, pairwise

    def grad_f_x(self, X):
        '''
        returns shape batch x length x num aa
        '''
        return (-((self.W(torch.flatten(X, 1, 2)) + self.h))).reshape(
            -1, self.L, self.A) / self.temp

    def neg_energy_and_grad_f_x(self, X, beta=1.0):
        '''
        returns shape batch x length x num aa
        '''
        tmp = self.W(torch.flatten(X, 1, 2))
        grad_f_x = -((tmp * beta + self.h))
        neg_energy = -X * (
            (tmp * beta / 2.0) + self.h).reshape(-1, self.L, self.A)
        neg_energy = neg_energy.reshape(-1, self.L * self.A).sum(dim=1)
        grad_f_x = grad_f_x.reshape(-1, self.L, self.A)
        return neg_energy / self.temp, grad_f_x / self.temp


def to_wt_gauge(h, J, wt_ind):
    '''
    Thank you Akosua!!!!
    '''
    assert (J.ndim == 4)
    assert (h.ndim == 2)
    select_positions = torch.arange(len(wt_ind))
    J_ij_ab = J
    J_ij_ci_b = J[select_positions, :, wt_ind].unsqueeze(2)
    J_ij_a_cj = J.transpose(-1, -2)[:, select_positions, wt_ind].unsqueeze(3)
    J_ij_ci_cj = J[select_positions, :,
                   wt_ind][:, select_positions,
                           wt_ind].unsqueeze(2).unsqueeze(3)
    J_new = J_ij_ab - J_ij_ci_b - J_ij_a_cj + J_ij_ci_cj

    h_i_c = h[select_positions, wt_ind].unsqueeze(1)
    J_j_nequal_i = (
        J.transpose(-1, -2)[:, select_positions, wt_ind].unsqueeze(3) -
        J[select_positions, :, wt_ind][:, select_positions,
                                       wt_ind].unsqueeze(2).unsqueeze(3)).sum(
                                           dim=1).squeeze()
    J_j_equal_i = (
        J.transpose(-1, -2)[:, select_positions, wt_ind].unsqueeze(3) -
        J[select_positions, :, wt_ind][:, select_positions,
                                       wt_ind].unsqueeze(2).unsqueeze(3)
    )[select_positions, select_positions].squeeze()
    h_new = (h - h_i_c + (J_j_nequal_i - J_j_equal_i)).to(torch.float)
    return h_new, J_new


def to_zs_gauge(h, J):
    assert (J.ndim == 4)
    assert (h.ndim == 2)
    L, A = h.shape
    J_zs = (J - J.mean(dim=2).unsqueeze(dim=2) -
            J.mean(dim=3).unsqueeze(dim=3) +
            (J.reshape(L, L, A * A).mean(dim=-1).unsqueeze(2).unsqueeze(3)))
    tmp = (
        J.mean(dim=3).unsqueeze(dim=3) -
        (J.reshape(L, L, A * A).mean(dim=-1).unsqueeze(2).unsqueeze(3))).sum(
            dim=1).squeeze()
    h_zs = (h - h.mean(dim=-1).unsqueeze(1) + tmp)
    return h_zs, J_zs
