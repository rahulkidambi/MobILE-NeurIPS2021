import os
from tqdm import tqdm

import numpy as np
import gym
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import time
import multiprocessing
import pdb

class DynamicsEnsemble:

    def __init__(self,
                 env_name,
                 n_models,
                 dataset,
                 threshold_const=1.0,
                 quantile_val=1.0,
                 batch_size=256,
                 hidden_sizes=[512, 512],
                 activation='relu',
                 transform=True,
                 optim_args={'optim':'sgd', 'lr':1e-4, 'momentum':0.9},
                 device=torch.device('cpu'),
                 base_seed=100,
                 env_dims=None,
                 num_cpus=2):

        self.transform = transform
        self.env_name=env_name
        self.device=device
        self.base_seed=base_seed
        self.n_model=n_models

        # Create the environment to check stats
        if env_name is None and env_dims is None:
            raise Exception("You either need to pass in environment id or env dimensions")
        if env_dims is None:
            env = gym.make(env_name)
            self.state_dim = env.observation_space.shape[0]
            if isinstance(env.action_space, Discrete):
                self.action_dim = env.action_space.n
                self.is_discrete = True
            elif isinstance(env.action_space, Box):
                self.action_dim = env.action_space.shape[0]
                self.is_discrete = False
            else:
                raise NotImplementedError("Action Space not yet supported")
            del env
        else:
            self.state_dim, self.action_dim = env_dims
        self.models = [DynamicsModel(self.state_dim,
                                     self.action_dim,
                                     hidden_sizes=hidden_sizes,
                                     activation=activation,
                                     transform=transform,
                                     optim_args=optim_args,
                                     device=device,
                                     seed=base_seed+k) for k in range(n_models)]

        self.dataset = dataset
        self.num_cpus = num_cpus
        self.batch_size = batch_size

        # Discrepancy Threshold
        self.threshold = 0.0 # Initially all (s,a) are unknown
        self.threshold_const = threshold_const
        self.quantile_val = quantile_val

    def add_data(self, state, action, next_state):
        self.dataset.add_data(state, action, next_state)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_cpus)
        self.transformations = self.dataset.get_transformations() if self.transform else None

    def train(self, n_epochs, logger=None, log_epoch=False, grad_clip=0.0, save_path=None):
        model_train_info = []
        for i, model in enumerate(self.models):
            if logger is not None:
                logger.info(f">>>> Training Model {i+1}/{self.n_model}")
            info = model.train_model(self.dataloader, n_epochs, transformations=self.transformations, \
                        logger=logger, model_num=i, log_epoch=log_epoch, grad_clip=grad_clip, save_path=save_path)
            model_train_info.append(info)
        return model_train_info

    def save(self, save_path):
        state_dicts = [model.state_dict() for model in self.models]
        torch.save({'models': state_dicts}, save_path)

    def load(self, model_path):
        state_dicts = torch.load(model_path, map_location=self.device)['models']
        if len(state_dicts) != len(self.models):
            raise Exception("The number of saved model weights does not equal number of models in ensemble")
        for model, state_dict in zip(self.models, state_dicts):
            model.load_state_dict(state_dict)
        # Transformations
        if self.transform:
            for model in self.models:
                model.state_mean, model.state_scale, model.action_mean, model.action_scale, \
                    model.diff_mean, model.diff_scale = self.transformations

    def compute_discrepancy(self, state, action):
        with torch.no_grad():
            preds = torch.cat([model.forward(state, action).unsqueeze(0) for model in self.models], dim=0)
        disc = torch.cat([torch.norm(preds[i]-preds[j], p=2, dim=1).unsqueeze(0) \
                   for i in range(preds.shape[0]) for j in range(i+1,preds.shape[0])], dim=0) # (n_pairs*batch)
        return disc.max(0).values.to(torch.device('cpu'))

    def compute_threshold(self):
        '''
        re-compute threshold when
        '''
        results = []
        for state, action, _ in self.dataloader:
            results.append(self.compute_discrepancy(state, action))
        results = torch.cat(results, dim=0)
        self.threshold = torch.quantile(results, self.quantile_val).item() * self.threshold_const
        self.all_discs = results

    def get_action_discrepancy(self, state, action):
        # Add Batch Dimension
        if len(state.shape) == 1: state.unsqueeze(0)
        if len(action.shape) == 1: action.unsqueeze(0)

        # One-hot if Discrete action space
        if self.is_discrete:
            action = torch.eye(self.action_dim)[action].view(action.shape[0], self.action_dim)

        state = state.float().to(self.device)
        action = action.float().to(self.device)
        return self.compute_discrepancy(state, action)

class DynamicsModel(nn.Module):

    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_sizes=[64, 64],
                 activation='relu',
                 transform=True,
                 optim_args={'optim':'sgd', 'lr':1e-4, 'momentum':0.9},
                 device=torch.device('cpu'),
                 seed = 100):
        super(DynamicsModel,self).__init__()

        # Set Seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.transform = transform

        # Define Model (NOTE: Currently supports only relu/tanh)
        non_linearity = nn.ReLU() if activation == 'relu' else nn.Tanh()
        layer_sizes = [state_dim+action_dim,] + hidden_sizes + [state_dim,]
        layers = []
        layers.append(nn.Linear(layer_sizes[0], layer_sizes[1]))
        for i in range(1,len(layer_sizes)-1):
            layers.append(non_linearity)
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.model = nn.Sequential(*layers).to(self.device)

        # Define Loss and Optimizer
        self.loss_fn = nn.MSELoss().to(self.device)
        if optim_args['optim'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=optim_args['lr'], \
                                             momentum=optim_args['momentum'], nesterov=True)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=optim_args['lr'], eps=optim_args['eps'])

    def forward(self, state, action, transform_out=True):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()

        state, action = state.to(self.device), action.to(self.device)
        if self.transform:
            state = (state - self.state_mean)/(self.state_scale)
            action = (action - self.action_mean)/(self.action_scale)

        state_diff = self.model.forward(torch.cat([state, action], dim=1))
        if self.transform and transform_out:
            state_diff = (state_diff * (self.diff_scale)) + self.diff_mean
        return state_diff

    # TODO: Move to a torch utils for NNs
    def get_grad_norms(self):
        params = [p for p in self.parameters() if p.grad is not None]
        if len(params) == 0:
            return 0
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in params]), 2)
        return total_norm.detach().cpu().item()

    def train_model(self,
                    dataloader,
                    n_epochs,
                    transformations=None,
                    logger=None,
                    model_num=None,
                    log_epoch=False,
                    grad_clip=0.0,
                    save_path=None):
        start_time = time.time()

        # Set Training
        self.model.train()

        # Transformations
        if self.transform:
            assert(transformations is not None)
            self.state_mean, self.state_scale, self.action_mean, self.action_scale, \
                self.diff_mean, self.diff_scale = transformations

        min_loss, min_epoch, max_grad = float('inf'), float('inf'), -float('inf')
        model_state_dict, optim_state_dict = None, None
        losses, grad_norms = [], []

        # Start Training Loop 
        for epoch in tqdm(range(n_epochs)):
            train_loss, epoch_grad_norm = [], []
            for state, action, next_state in dataloader:
                self.optimizer.zero_grad()
                state = state.to(self.device)
                action = action.to(self.device)
                target = (next_state - state).to(self.device)
                pred = self.forward(state, action, transform_out=False)
                if self.transform:
                    target = (target - self.diff_mean)/(self.diff_scale)
                target = target.to(self.device)
                loss = self.loss_fn(pred, target)
                loss.backward()

                # Clip Gradient Norm
                epoch_grad_norm.append(self.get_grad_norms())
                if grad_clip:
                    nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

                self.optimizer.step()
                train_loss.append(loss.item())

            # Per Epoch Boilerplate
            loss = sum(train_loss)/len(train_loss)
            if loss < min_loss:
                min_epoch = epoch
                min_loss = loss
                model_state_dict = self.state_dict()
                optim_state_dict = self.optimizer.state_dict()

            # Save Checkpoints
            if save_path is not None:
                if (epoch+1) % 100 == 0 or (epoch+1) == n_epochs/2:
                    torch.save({'model': self.state_dict(),
                                'optim': self.optimizer.state_dict(),
                                'epoch': epoch+1,
                                'loss' : loss}, os.path.join(save_path, f'{epoch+1}_checkpoint.pt'))

            # Store Gradient Norms
            grad_norms += epoch_grad_norm
            curr_grad_avg = sum(epoch_grad_norm)/len(epoch_grad_norm)
            if curr_grad_avg > max_grad:
                max_grad = curr_grad_avg
            if logger is not None:
                losses.append(loss)
                if log_epoch: logger.info('Epoch {} Loss: {}'.format(epoch, loss))

        # Load in the minimum states
        self.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optim_state_dict)

        if save_path is not None:
            torch.save({'model': self.state_dict(),
                        'optim': self.optimizer.state_dict(),
                        'epoch': min_epoch+1,
                        'loss' : min_loss}, os.path.join(save_path, 'best_checkpoint.pt'))

        if logger is not None:
            if model_num is not None:
                logger.info('Dynamics Model {} Start | Best Loss: {} | {}'.format(model_num, losses[0], min_loss))
            else:
                logger.info('Dynamics Model Start | Best Loss: {} | {}'.format(losses[0], min_loss))
        return min_loss, losses[0], grad_norms
