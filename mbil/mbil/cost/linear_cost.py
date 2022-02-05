import torch
import torch.nn as nn

import numpy as np
import pdb

UPDATE_TYPES = ['exact', 'geometric', 'decay', 'decay_sqrt', 'ficticious', 'gd', 'polynomial', 'exponential']

class RBFLinearCost:
    """
    MMD cost implementation with rff feature representations

    TODO: Currently hardcoded to cpu....ok for now

    :param expert_data: (torch Tensor) expert data used for feature matching
    :param feature_dim: (int) feature dimension for rff
    :param input_type: (str) state (s), state-action (sa), state-next state (ss),
                       state-action-next state (sas)
    :param update_type: (str) exact, geometric, decay, decay_sqrt, ficticious
    :param cost_range: (list) inclusive range of costs
    :param bw_quantile: (float) quantile used to fit bandwidth for rff kernel
    :param bw_samples: (int) number of samples used to fit bandwidth
    :param lambda: (float) weight parameter for bonus and cost
    :param lr: (float) learning rate for discriminator/cost update. 0.0 = closed form update
    :param seed: (int) random seed to set cost function
    """
    def __init__(self,
                 expert_data,
                 feature_dim=1024,
                 input_type='s',
                 update_type='exact',
                 cost_range=[-1.,0.],
                 bw_quantile=0.1,
                 bw_samples=100000,
                 lambda_b=1.0,
                 lr=0.0,
                 T=400,
                 seed=100):

        # Set Random Seed 
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.expert_data = expert_data
        input_dim = expert_data.size(1)
        self.input_type = input_type
        self.feature_dim = feature_dim
        self.cost_range = cost_range
        if cost_range is not None:
            self.c_min, self.c_max = cost_range
        self.lambda_b = lambda_b
        self.lr = lr

        # Fit Bandwidth
        self.quantile = bw_quantile
        self.bw_samples = bw_samples
        self.bw = self.fit_bandwidth(expert_data)
        self.expert_data = expert_data

        # Define Phi and Cost weights
        self.rff = nn.Linear(input_dim, feature_dim)
        self.rff.bias.data = (torch.rand_like(self.rff.bias.data)-0.5)*2.0*np.pi
        self.rff.weight.data = torch.rand_like(self.rff.weight.data)/(self.bw+1e-8)

        # W Update Init
        self.T = T
        self.w_bar = None
        self.w = None
        self.w_ctr, self.w_sum = 0, 0
        self.update_type = update_type
        if self.update_type not in UPDATE_TYPES:
            raise NotImplementedError("This update type is not available")

        # Compute Expert Phi Mean
        self.expert_rep = self.get_rep(expert_data)
        self.phi_e = self.expert_rep.mean(dim=0)

    def get_rep(self, x):
        with torch.no_grad():
            out = self.rff(x.cpu())
            out = torch.cos(out)*np.sqrt(2/self.feature_dim)
        return out

    def update_bandwidth(self, buffer):
        #data = buffer.state
        data = torch.from_numpy(buffer).float()
        combined_data = torch.cat([self.expert_data, data], dim=0)
        self.bw = self.fit_bandwidth(combined_data)

    def fit_bandwidth(self, data):
        num_data = data.shape[0]
        idxs_0 = torch.randint(low=0, high=num_data, size=(self.bw_samples,))
        idxs_1 = torch.randint(low=0, high=num_data, size=(self.bw_samples,))
        norm = torch.norm(data[idxs_0, :]-data[idxs_1, :], dim=1)
        bw = torch.quantile(norm, q=self.quantile).item()
        return bw

    def fit_cost(self, data_pi):
        phi = self.get_rep(data_pi).mean(0)
        feat_diff = phi - self.phi_e

        # Closed form solution
        if self.update_type == "exact":
            self.w = feat_diff

        # alpha * old_w + (1-alpha) * new_w
        if self.update_type == "geometric":
            self.w = feat_diff if self.w is None else \
                self.lr*self.w + (1-self.lr)*feat_diff

        # (1- 1/t) * old_w + (1/t) * new_w
        if self.update_type == "decay" or self.update_type == "decay_sqrt":
            self.w_ctr += 1
            t = self.w_ctr if self.update_type == "decay" else np.sqrt(self.w_ctr)
            self.w_sum = 1 if self.w is None else (1-1/t)*self.w_sum + 1/t
            self.w_bar = feat_diff if self.w_bar is None else \
                (1-1/t)*self.w_bar + (1/t)*self.w
            self.w = self.w_bar/self.w_sum

        # average all w
        if self.update_type == "ficticious":
            self.w_ctr += 1
            self.w_bar = feat_diff if self.w_bar is None else self.w_bar + feat_diff
            self.w = self.w_bar/self.w_ctr

        if self.update_type == 'gd':
            self.w = self.w + self.lr*feat_diff if self.w is not None else self.lr*feat_diff

        if self.update_type == 'polynomial':
            self.w_ctr += 1
            self.w = (1-(1/self.w_ctr))*self.w + feat_diff if self.w is not None else feat_diff

        if self.update_type == 'exponential':
            self.w_ctr += 1
            self.w = (1-(self.w_ctr/self.T))*self.w + feat_diff if self.w is not None else feat_diff

        return torch.dot(self.w, feat_diff).item()

    def get_costs(self, x):
        data = self.get_rep(x)
        if self.cost_range is not None:
            return torch.clamp(torch.mm(data, self.w.unsqueeze(1)), self.c_min, self.c_max)
        return torch.mm(data, self.w.unsqueeze(1))

    def get_expert_cost(self):
        return (1-self.lambda_b)*torch.clamp(torch.mm(self.expert_rep, self.w.unsqueeze(1)), self.c_min, self.c_max).mean()

    def get_bonus_costs(self, states, actions, ensemble, next_states=None):
        if self.input_type == 'sa':
            rff_input = torch.cat([states, actions], dim=1)
        elif self.input_type == 'ss':
            assert(next_states is not None)
            rff_input = torch.cat([states, next_states], dim=1)
        elif self.input_type == 'sas':
            rff_input = torch.cat([states, actions, next_states], dim=1)
        elif self.input_type == 's':
            rff_input = states
        else:
            raise NotImplementedError("Input type not implemented")

        # Get Linear Cost 
        rff_cost = self.get_costs(rff_input)

        if self.cost_range is not None:
            # Get Bonus from Ensemble
            discrepancy = ensemble.get_action_discrepancy(states, actions)/ensemble.threshold
            discrepancy = discrepancy.view(-1, 1)
            #discrepancy[discrepancy>1.0] = 1.0
            # Bonus is LOW if (s,a) is unknown
            bonus = discrepancy * self.c_min
        else:
            bonus = ensemble.get_action_discrepancy(states, actions).view(-1,1)

        # Weight cost components
        ipm = (1-self.lambda_b)*rff_cost
        weighted_bonus = self.lambda_b*bonus.cpu() # Note cpu hardcoding

        # Optimism
        cost = ipm + weighted_bonus

        # Logging info
        info = {'bonus': weighted_bonus, 'ipm': ipm, 'v_targ': rff_cost, 'cost': cost}

        return cost, info

