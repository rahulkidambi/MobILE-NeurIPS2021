import numpy as np
import numpy.linalg as la
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import math
import itertools
import pdb

class DynamicsModel(nn.Module):

    def __init__(self, state_dim, act_dim, hidden_sizes = [64,32], seed = 100, fit_lr=1e-4, fit_wd=0.0, device='cpu', activation='relu', transform=True, momentum = 0.9, isSGD = True):
        super(DynamicsModel,self).__init__()

        #torch.manual_seed(seed)
        #np.random.seed(seed)
        #if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        self.state_dim, self.act_dim, self.seed, self.std = state_dim, act_dim, seed, None
        self.fit_lr, self.fit_wd, self.momentum, self.isSGD  = fit_lr, fit_wd, momentum, isSGD
        self.device = 'cuda' if device == 'gpu' or device == 'cuda' else 'cpu'
        self.loss_fn = nn.MSELoss().to(self.device)
        self.non_linearity = nn.ReLU() if activation == 'relu' else nn.Tanh()
        self.transform = transform
        all_layers = [state_dim+act_dim, ] + hidden_sizes + [state_dim,]
        self.layers = []
        self.layers.append( nn.Linear(all_layers[0], all_layers[1]) )
        for i in range(1,len(all_layers)-1):
            self.layers.append( self.non_linearity )
            self.layers.append( nn.Linear(all_layers[i], all_layers[i+1]) )
        self.model = nn.Sequential(*self.layers).to(self.device)
        #self.model = MlpDynamics(state_dim,act_dim)
        if self.isSGD:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.fit_lr, weight_decay=self.fit_wd, momentum=self.momentum, nesterov=True)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.fit_lr, weight_decay=self.fit_wd)
        #self.layers = []
        #self.layers.append( nn.Linear(all_layers[0], all_layers[1]) )
        #for i in range(1,len(all_layers)-1):
        #    self.layers.append( self.non_linearity )
        #    self.layers.append( nn.Linear(all_layers[i], all_layers[i+1]) )
        # self.model = nn.Sequential(*self.layers)
        #self.model = MlpDynamics(state_dim,act_dim,state_hidden_sizes=[1024,1024],action_hidden_sizes=[512,512],head_sizes=[512,512])
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.fit_lr, weight_decay=1e-4)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 800, gamma=0.1)

    def reinit_optimizer(self):
        if self.isSGD:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.fit_lr, weight_decay=self.fit_wd, momentum=self.momentum, nesterov=True)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.fit_lr, weight_decay=self.fit_wd)

    def compute_dataset_stats(self, dataset):
        self.state_mean, self.state_scale, self.diff_mean, self.diff_scale = dataset.get_transformations()

    def refresh_model(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(self.seed)
        self.model = nn.Sequential(*self.layers)
        if self.isSGD:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.fit_lr, weight_decay=self.fit_wd, momentum=self.momentum, nesterov=True)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.fit_lr, weight_decay=self.fit_wd)

    def forward(self, s, a, no_grad=True, transform=True):
        if type(s) == np.ndarray:
            s = torch.from_numpy(s).float()
        if type(a) == np.ndarray:
            a = torch.from_numpy(a).float()

        s ,a = s.to(self.device), a.to(self.device)
        if no_grad:
            if self.transform and transform:
                s = (s - self.state_mean)/(self.state_scale)
                a = (a - self.action_mean)/(self.action_scale)
            with torch.no_grad():
                s_new = self.model.forward(torch.cat([s.float(), a.float()], dim=1)) # Note: a is concatenation of one hots
                #s_new = self.model.forward(s.float(), a.float()) # Note: a is concatenation of one hots
            if self.transform and transform:
                s_new = (s_new * (self.diff_scale)) + self.diff_mean
        else:
            #s_new = self.model.forward(s.float(), a.float())
            s_new = self.model.forward(torch.cat([s, a], dim=1)) # Note: a is the concatenation of one hots
        return s_new

    def predict(self, s, a):
        return self.forward(s,a).to('cpu').data.numpy()
    
    def get_grad_norms(self):
        params = [p for p in self.parameters() if p.grad is not None]
        if len(params) == 0:
            return 0
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in params]), 2)
        return total_norm.detach().cpu().item()

    def fit_std(self,dataset):
        self.model.eval()
        for i,x in enumerate(dataset.a):
            if i==0:
                a_ = dataset.one_hot_action(x).unsqueeze(0)
            else:
                a_ = torch.cat([a_, dataset.one_hot_action(x).unsqueeze(0)],0)
        dataset.s, dataset.sp, a_ = dataset.s.to(self.device), dataset.sp.to(self.device), a_.to(self.device)
        st = time.time()
        num_samples = dataset.s.shape[0]
        differences = []
        for data_idx in range(num_samples):
            s, sp, a = dataset.s[data_idx], dataset.sp[data_idx], a_[data_idx]
            # State Norm
            if self.transform:
                s = (s-self.state_mean)/self.state_scale
                a = (a-self.action_mean)/self.action_scale
            s, a = s.float().to(self.device), a.float().to(self.device)
            pred = self.forward(s.unsqueeze(0), a.unsqueeze(0), no_grad=False) # not doing self.forward
            if self.transform:
                pred = (pred * (self.diff_scale)) + self.diff_mean
            pred += dataset.s[data_idx]# next state prediction
            differences.append( (pred.squeeze(0)-sp).detach().cpu().numpy() )
        dataset.s, dataset.sp, a_ = dataset.s.to('cpu'), dataset.sp.to('cpu'), a_.to('cpu')
        if self.std is not None:
            self.std = 0.999 * self.std + 0.001*np.array(differences).std(axis=0)
        else:
            self.std = np.array(differences).std(axis=0)
        print(self.std)

    def train_model(self,
                    dataset,
                    n_epochs,
                    logger=None,
                    model_num=None,
                    log_epoch=False,
                    batch_size=64,
                    reinit_optimizer=False,
                    grad_clip=0.0,
                    log_grad_norm=False):
        if reinit_optimizer is True: self.reinit_optimizer()
        # Transformations
        if self.transform:
            self.state_mean, self.state_scale, self.action_mean, self.action_scale, self.diff_mean, self.diff_scale = dataset.get_transformations()
        self.model.train()
        for i,x in enumerate(dataset.a):
            if i==0:
                a_ = dataset.one_hot_action(x).unsqueeze(0)
            else:
                a_ = torch.cat([a_, dataset.one_hot_action(x).unsqueeze(0)],0)
        #dloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
        dataset.s, dataset.sp, a_ = dataset.s.to(self.device), dataset.sp.to(self.device), a_.to(self.device)
        st = time.time()
        losses = []
        num_samples = dataset.s.shape[0]
        min_loss = float('inf')
        max_grad = -float('inf')
        model_state_dict, optim_state_dict = None, None
        grad_norms = []
        for e in range(n_epochs):
            rand_idx = torch.LongTensor(np.random.permutation(num_samples)).to(self.device)
            train_loss, n_b = 0., 0
            num_batches = int(num_samples//batch_size) + 1 * (num_samples%batch_size != 0)
            if log_grad_norm: epoch_grad_norm = []
            for mb in range(num_batches):
            #for _,(s,a,sp) in enumerate(dloader):
                data_idx = rand_idx[mb*batch_size:min((mb+1)*batch_size,num_samples)]
                s, sp, a = dataset.s[data_idx], dataset.sp[data_idx], a_[data_idx]
                # State Norm
                target = (sp - s).float()
                if self.transform:
                    s = (s-self.state_mean)/self.state_scale
                    a = (a-self.action_mean)/self.action_scale
                s, a = s.float().to(self.device), a.float().to(self.device)
                self.optimizer.zero_grad()
                pred = self.forward(s, a, no_grad=False) # not doing self.forward
                if self.transform:
                    target = (target - self.diff_mean)/(self.diff_scale)
                target = target.to(self.device)
                loss = self.loss_fn(pred, target)
                loss.backward()
                # Clip Gradient Norm
                if log_grad_norm:
                    epoch_grad_norm.append(self.get_grad_norms())
                if grad_clip:
                    nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                self.optimizer.step()
                train_loss += loss.item()
                n_b += 1
            # self.scheduler.step()
            loss = train_loss/n_b
            if loss < min_loss:
                min_loss = loss
                model_state_dict = self.state_dict()
                optim_state_dict = self.optimizer.state_dict()
            if log_grad_norm:
                grad_norms += epoch_grad_norm
                curr_grad_avg = sum(epoch_grad_norm)/len(epoch_grad_norm)
                if curr_grad_avg > max_grad:
                    max_grad = curr_grad_avg
            if logger is not None:
                losses.append(loss)
                if log_epoch:
                    logger.info('Epoch {} Loss: {}'.format(e, loss))
        dataset.s, dataset.sp, a_ = dataset.s.to('cpu'), dataset.sp.to('cpu'), a_.to('cpu')

        # Load in the minimum states
        self.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optim_state_dict)
        self.fit_std(dataset)
        if logger is not None:
            if model_num is not None:
                logger.info('Dynamics Model {} Start | Best Loss: {} | {}'.format(model_num, losses[0], min_loss))
            else:
                logger.info('Dynamics Model Start | Best Loss: {} | {}'.format(losses[0], min_loss))
        if log_grad_norm:
            return min_loss, losses[0], grad_norms
        return min_loss, losses[0]

class DynamicsModelContinuous(nn.Module):

    def __init__(self, state_dim, act_dim, hidden_sizes = [64,32], seed = 100, fit_lr=1e-4, fit_wd=0.0, device='cpu', activation='relu', transform=True, momentum = 0.9, isSGD = True):
        super(DynamicsModelContinuous, self).__init__()

        self.state_dim, self.act_dim, self.seed = state_dim, act_dim, seed
        self.fit_lr, self.fit_wd, self.momentum, self.isSGD  = fit_lr, fit_wd, momentum, isSGD
        self.device = 'cuda' if device == 'gpu' or device == 'cuda' else 'cpu'
        self.loss_fn = nn.MSELoss().to(self.device)
        self.non_linearity = nn.ReLU() if activation == 'relu' else nn.Tanh()
        self.transform = transform
        all_layers = [state_dim+act_dim, ] + hidden_sizes + [state_dim,]
        self.layers = []
        self.layers.append( nn.Linear(all_layers[0], all_layers[1]) )
        for i in range(1,len(all_layers)-1):
            self.layers.append( self.non_linearity )
            self.layers.append( nn.Linear(all_layers[i], all_layers[i+1]) )
        self.model = nn.Sequential(*self.layers).to(self.device)
        if self.isSGD:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.fit_lr, weight_decay=self.fit_wd, momentum=self.momentum, nesterov=True)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.fit_lr, weight_decay=self.fit_wd)

    def reinit_optimizer(self):
        if self.isSGD:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.fit_lr, weight_decay=self.fit_wd, momentum=self.momentum, nesterov=True)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.fit_lr, weight_decay=self.fit_wd)

    def compute_dataset_stats(self, dataset):
        self.state_mean, self.state_scale, self.diff_mean, self.diff_scale = dataset.get_transformations()

    def refresh_model(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(self.seed)
        self.model = nn.Sequential(*self.layers)
        if self.isSGD:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.fit_lr, weight_decay=self.fit_wd, momentum=self.momentum, nesterov=True)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.fit_lr, weight_decay=self.fit_wd)

    def forward(self, s, a, no_grad=True, transform=True):
        if type(s) == np.ndarray:
            s = torch.from_numpy(s).float()
        if type(a) == np.ndarray:
            a = torch.from_numpy(a).float()

        s ,a = s.to(self.device), a.to(self.device)
        if no_grad:
            if self.transform and transform:
                s = (s - self.state_mean)/(self.state_scale)
                a = (a - self.action_mean)/(self.action_scale)
            with torch.no_grad():
                s_new = self.model.forward(torch.cat([s.float(), a.float()], dim=1)) 
            if self.transform and transform:
                s_new = (s_new * (self.diff_scale)) + self.diff_mean
        else:
            s_new = self.model.forward(torch.cat([s, a], dim=1))
        return s_new

    def predict(self, s, a):
        return self.forward(s,a).to('cpu').data.numpy()
    
    def get_grad_norms(self):
        params = [p for p in self.parameters() if p.grad is not None]
        if len(params) == 0:
            return 0
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in params]), 2)
        return total_norm.detach().cpu().item()

    def train_model(self,
                    dataset,
                    n_epochs,
                    logger=None,
                    model_num=None,
                    log_epoch=False,
                    batch_size=64,
                    reinit_optimizer=False,
                    grad_clip=0.0,
                    log_grad_norm=False):
        if reinit_optimizer is True: self.reinit_optimizer()
        # Transformations
        if self.transform:
            self.state_mean, self.state_scale, self.action_mean, self.action_scale, self.diff_mean, self.diff_scale = dataset.get_transformations()
        self.model.train()
        dataset.s, dataset.sp, dataset.a = dataset.s.to(self.device), dataset.sp.to(self.device), dataset.a.to(self.device)
        st = time.time()
        losses = []
        num_samples = dataset.s.shape[0]
        min_loss = float('inf')
        max_grad = -float('inf')
        model_state_dict, optim_state_dict = None, None
        grad_norms = []
        for e in range(n_epochs):
            rand_idx = torch.LongTensor(np.random.permutation(num_samples)).to(self.device)
            train_loss, n_b = 0., 0
            num_batches = int(num_samples//batch_size) + 1 * (num_samples%batch_size != 0)
            if log_grad_norm: epoch_grad_norm = []
            for mb in range(num_batches):
            #for _,(s,a,sp) in enumerate(dloader):
                data_idx = rand_idx[mb*batch_size:min((mb+1)*batch_size,num_samples)]
                s, sp, a = dataset.s[data_idx], dataset.sp[data_idx], dataset.a[data_idx]
                # State Norm
                target = (sp - s).float()
                if self.transform:
                    s = (s-self.state_mean)/self.state_scale
                    a = (a-self.action_mean)/self.action_scale
                s, a = s.float().to(self.device), a.float().to(self.device)
                self.optimizer.zero_grad()
                pred = self.forward(s, a, no_grad=False) # not doing self.forward
                if self.transform:
                    target = (target - self.diff_mean)/(self.diff_scale)
                target = target.to(self.device)
                loss = self.loss_fn(pred, target)
                loss.backward()
                # Clip Gradient Norm
                if log_grad_norm:
                    epoch_grad_norm.append(self.get_grad_norms())
                if grad_clip:
                    nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                self.optimizer.step()
                train_loss += loss.item()
                n_b += 1
            # self.scheduler.step()
            loss = train_loss/n_b
            if loss < min_loss:
                min_loss = loss
                model_state_dict = self.state_dict()
                optim_state_dict = self.optimizer.state_dict()
            if log_grad_norm:
                grad_norms += epoch_grad_norm
                curr_grad_avg = sum(epoch_grad_norm)/len(epoch_grad_norm)
                if curr_grad_avg > max_grad:
                    max_grad = curr_grad_avg
            if logger is not None:
                losses.append(loss)
                if log_epoch:
                    logger.info('Epoch {} Loss: {}'.format(e, loss))
        dataset.s, dataset.sp, dataset.a = dataset.s.to('cpu'), dataset.sp.to('cpu'), dataset.a.to('cpu')

        # Load in the minimum states
        self.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optim_state_dict)

        if logger is not None:
            if model_num is not None:
                logger.info('Dynamics Model {} Start | Best Loss: {} | {}'.format(model_num, losses[0], min_loss))
            else:
                logger.info('Dynamics Model Start | Best Loss: {} | {}'.format(losses[0], min_loss))
        if log_grad_norm:
            return min_loss, losses[0], grad_norms
        return min_loss, losses[0]

def layer_init(layer, w_scale=1.0, bias_std=None):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    if not bias_std:
        nn.init.constant_(layer.bias.data, 0)
    if bias_std:
        nn.init.normal_(layer.bias.data, mean=0.0, std=bias_std)
    return layer

class MlpDynamics(nn.Module):
    def __init__(self, state_dim, act_dim, state_hidden_sizes = [64,64], action_hidden_sizes = [32,16], head_sizes = [32,32]):
        super(MlpDynamics, self).__init__()

        all_layers = [state_dim,] + state_hidden_sizes # + [state_dim,]
        self.non_linearity = nn.ReLU()
        self.state_layers = []
        layer = nn.Linear(all_layers[0], all_layers[1])
        layer_init(layer)
        self.state_layers.append( layer )
        for i in range(1,len(all_layers)-1):
            self.state_layers.append( self.non_linearity )
            layer = nn.Linear(all_layers[i], all_layers[i+1])
            layer_init(layer)
            self.state_layers.append( layer )
        self.state_encoder = nn.Sequential(*self.state_layers)

        all_layers = [act_dim,] + action_hidden_sizes
        self.action_layers = []
        layer = nn.Linear(all_layers[0], all_layers[1])
        layer_init(layer)
        self.action_layers.append( layer )
        for i in range(1,len(all_layers)-1):
            self.action_layers.append( self.non_linearity )
            layer = nn.Linear(all_layers[i], all_layers[i+1])
            self.action_layers.append( layer )
        self.action_encoder = nn.Sequential(*self.action_layers)

        all_layers = [state_hidden_sizes[-1]+action_hidden_sizes[-1], ] + head_sizes + [state_dim,]
        self.head_layers = []
        layer = nn.Linear(all_layers[0], all_layers[1])
        layer_init(layer)
        self.head_layers.append( layer )
        for i in range(1,len(all_layers)-1):
            self.head_layers.append( self.non_linearity )
            layer = nn.Linear(all_layers[i], all_layers[i+1])
            self.head_layers.append( layer )
        self.head = nn.Sequential(*self.head_layers)

        #self.f1 = nn.Linear(state_dim, hidden_sizes[0])
        #self.f2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        #self.f3 = nn.Linear(hidden_sizes[1]+10, hidden_sizes[1])
        #self.f4 = nn.Linear(hidden_sizes[1], state_dim)

        #self.a1 = nn.Linear(act_dim, 10)
        #self.a2 = nn.Linear(10, 10)

        # Initialize layers
        # layer_init(self.f1)
        # layer_init(self.f2)
        # layer_init(self.f3)
        # layer_init(self.f4)

        #layer_init(self.a1)
        #layer_init(self.a2)

    def forward(self, s, a):
        state_enc = self.state_encoder(s)
        action_enc = self.action_encoder(a)
        return self.head(torch.cat([state_enc,action_enc], dim=1))
        # x = torch.relu(self.f1(s))
        # x = torch.relu(self.f2(x))

        # enc_a = torch.relu(self.a1(a))
        # enc_a = torch.relu(self.a2(enc_a))

        x = torch.relu(self.f3(torch.cat([x, enc_a], dim = 1)))
        x = self.f4(x)
        return x

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class WorldModel:
    def __init__(self, state_dim, act_dim,
                 learn_reward=False,
                 hidden_size=(64,64),
                 seed=123,
                 fit_lr=1e-3,
                 fit_wd=0.0,
                 device='cpu',
                 activation='relu',
                 residual=True,
                 *args,
                 **kwargs,):

        self.state_dim, self.act_dim = state_dim, act_dim
        self.device, self.learn_reward = device, learn_reward
        if self.device == 'gpu' : self.device = 'cuda'
        # construct the dynamics model
        self.dynamics_net = DynamicsNet(state_dim, act_dim, hidden_size, residual=residual, seed=seed).to(self.device)
        self.dynamics_net.set_transformations()  # in case device is different from default, it will set transforms correctly
        if activation == 'tanh' : self.dynamics_net.nonlinearity = torch.tanh
        self.dynamics_opt = torch.optim.Adam(self.dynamics_net.parameters(), lr=fit_lr, weight_decay=fit_wd)
        self.dynamics_loss = torch.nn.MSELoss()
        # construct the reward model if necessary
        if self.learn_reward:
            # small network for reward is sufficient if we augment the inputs with next state predictions
            self.reward_net = RewardNet(state_dim, act_dim, hidden_size=(100, 100), seed=seed).to(self.device)
            self.reward_net.set_transformations()  # in case device is different from default, it will set transforms correctly
            if activation == 'tanh' : self.reward_net.nonlinearity = torch.tanh
            self.reward_opt = torch.optim.Adam(self.reward_net.parameters(), lr=fit_lr, weight_decay=fit_wd)
            self.reward_loss = torch.nn.MSELoss()
        else:
            self.reward_net, self.reward_opt, self.reward_loss = None, None, None

    def to(self, device):
        self.dynamics_net.to(device)
        if self.learn_reward : self.reward_net.to(device)

    def is_cuda(self):
        return next(self.dynamics_net.parameters()).is_cuda

    def forward(self, s, a):
        if type(s) == np.ndarray:
            s = torch.from_numpy(s).float()
        if type(a) == np.ndarray:
            a = torch.from_numpy(a).float()
        s = s.to(self.device)
        a = a.to(self.device)
        return self.dynamics_net.forward(s, a)

    def predict(self, s, a):
        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).float()
        s = s.to(self.device)
        a = a.to(self.device)
        s_next = self.dynamics_net.forward(s, a)
        s_next = s_next.to('cpu').data.numpy()
        return s_next

    def reward(self, s, a):
        if not self.learn_reward:
            print("Reward model is not learned. Use the reward function from env.")
            return None
        else:
            if type(s) == np.ndarray:
                s = torch.from_numpy(s).float()
            if type(a) == np.ndarray:
                a = torch.from_numpy(a).float()
            s = s.to(self.device)
            a = a.to(self.device)
            sp = self.dynamics_net.forward(s, a).detach().clone()
            return self.reward_net.forward(s, a, sp)

    def compute_loss(self, s, a, s_next):
        # Intended for logging use only, not for loss computation
        sp = self.forward(s, a)
        s_next = torch.from_numpy(s_next).float() if type(s_next) == np.ndarray else s_next
        s_next = s_next.to(self.device)
        loss = self.dynamics_loss(sp, s_next)
        return loss.to('cpu').data.numpy()

    def fit_dynamics(self, s, a, sp, fit_mb_size, fit_epochs, max_steps=1e4, 
                     set_transformations=True, *args, **kwargs):
        # move data to correct devices
        assert type(s) == type(a) == type(sp)
        assert s.shape[0] == a.shape[0] == sp.shape[0]
        if type(s) == np.ndarray:
            s = torch.from_numpy(s).float()
            a = torch.from_numpy(a).float()
            sp = torch.from_numpy(sp).float()
        s = s.to(self.device); a = a.to(self.device); sp = sp.to(self.device)
       
        # set network transformations
        if set_transformations:
            s_shift, a_shift = torch.mean(s, dim=0), torch.mean(a, dim=0)
            s_scale, a_scale = torch.mean(torch.abs(s - s_shift), dim=0), torch.mean(torch.abs(a - a_shift), dim=0)
            out_shift = torch.mean(sp-s, dim=0) if self.dynamics_net.residual else torch.mean(sp, dim=0)
            out_scale = torch.mean(torch.abs(sp-s-out_shift), dim=0) if self.dynamics_net.residual else torch.mean(torch.abs(sp-out_shift), dim=0)
            self.dynamics_net.set_transformations(s_shift, s_scale, a_shift, a_scale, out_shift, out_scale)

        # prepare dataf for learning
        if self.dynamics_net.residual:  
            X = (s, a) ; Y = (sp - s - out_shift) / (out_scale + 1e-8)
        else:
            X = (s, a) ; Y = (sp - out_shift) / (out_scale + 1e-8)
        # disable output transformations to learn in the transformed space
        self.dynamics_net._apply_out_transforms = False
        return_vals =  fit_model(self.dynamics_net, X, Y, self.dynamics_opt, self.dynamics_loss,
                                 fit_mb_size, fit_epochs, max_steps=max_steps)
        self.dynamics_net._apply_out_transforms = True
        return return_vals

    def fit_reward(self, s, a, r, fit_mb_size, fit_epochs, max_steps=1e4, 
                   set_transformations=True, *args, **kwargs):
        if not self.learn_reward:
            print("Reward model was not initialized to be learnable. Use the reward function from env.")
            return None

        # move data to correct devices
        assert type(s) == type(a) == type(r)
        assert len(r.shape) == 2 and r.shape[1] == 1  # r should be a 2D tensor, i.e. shape (N, 1)
        assert s.shape[0] == a.shape[0] == r.shape[0]
        if type(s) == np.ndarray:
            s = torch.from_numpy(s).float()
            a = torch.from_numpy(a).float()
            r = torch.from_numpy(r).float()
        s = s.to(self.device); a = a.to(self.device); r = r.to(self.device)
       
        # set network transformations
        if set_transformations:
            s_shift, a_shift = torch.mean(s, dim=0), torch.mean(a, dim=0)
            s_scale, a_scale = torch.mean(torch.abs(s-s_shift), dim=0), torch.mean(torch.abs(a-a_shift), dim=0)
            r_shift, r_scale = torch.mean(r, dim=0), torch.mean(torch.abs(r-r_shift), dim=0)
            self.reward_net.set_transformations(s_shift, s_scale, a_shift, a_scale, r_shift, r_scale)

        # get next state prediction
        sp = self.dynamics_net.forward(s, a).detach().clone()

        # call the generic fit function
        X = (s, a, sp) ; Y = r
        return fit_model(self.reward_net, X, Y, self.reward_opt, self.reward_loss,
                         fit_mb_size, fit_epochs, max_steps=max_steps)

    def compute_path_rewards(self, paths):
        # paths has two keys: observations and actions
        # paths["observations"] : (num_traj, horizon, obs_dim)
        # paths["rewards"] should have shape (num_traj, horizon)
        if not self.learn_reward: 
            print("Reward model is not learned. Use the reward function from env.")
            return None
        s, a = paths['observations'], paths['actions']
        num_traj, horizon, s_dim = s.shape
        a_dim = a.shape[-1]
        s = s.reshape(-1, s_dim)
        a = a.reshape(-1, a_dim)
        r = self.reward(s, a)
        r = r.to('cpu').data.numpy().reshape(num_traj, horizon)
        paths['rewards'] = r


class DynamicsNet(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size=(64,64),
                 s_shift = None,
                 s_scale = None,
                 a_shift = None,
                 a_scale = None,
                 out_shift = None,
                 out_scale = None,
                 out_dim = None,
                 residual = True,
                 seed=123,
                 use_mask = True,
                 ):
        super(DynamicsNet, self).__init__()

        torch.manual_seed(seed)
        self.state_dim, self.act_dim, self.hidden_size = state_dim, act_dim, hidden_size
        self.out_dim = state_dim if out_dim is None else out_dim
        self.layer_sizes = (state_dim + act_dim, ) + hidden_size + (self.out_dim, )
        # hidden layers
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])
                                        for i in range(len(self.layer_sizes)-1)])
        self.nonlinearity = torch.relu
        self.residual, self.use_mask = residual, use_mask
        self._apply_out_transforms = True
        self.set_transformations(s_shift, s_scale, a_shift, a_scale, out_shift, out_scale)

    def set_transformations(self, s_shift=None, s_scale=None,
                            a_shift=None, a_scale=None,
                            out_shift=None, out_scale=None):

        if s_shift is None:
            self.s_shift     = torch.zeros(self.state_dim)
            self.s_scale    = torch.ones(self.state_dim)
            self.a_shift     = torch.zeros(self.act_dim)
            self.a_scale    = torch.ones(self.act_dim)
            self.out_shift   = torch.zeros(self.out_dim)
            self.out_scale  = torch.ones(self.out_dim)
        elif type(s_shift) == torch.Tensor:
            self.s_shift, self.s_scale = s_shift, s_scale
            self.a_shift, self.a_scale = a_shift, a_scale
            self.out_shift, self.out_scale = out_shift, out_scale
        elif type(s_shift) == np.ndarray:
            self.s_shift     = torch.from_numpy(np.float32(s_shift))
            self.s_scale    = torch.from_numpy(np.float32(s_scale))
            self.a_shift     = torch.from_numpy(np.float32(a_shift))
            self.a_scale    = torch.from_numpy(np.float32(a_scale))
            self.out_shift   = torch.from_numpy(np.float32(out_shift))
            self.out_scale  = torch.from_numpy(np.float32(out_scale))
        else:
            print("Unknown type for transformations")
            quit()

        device = next(self.parameters()).data.device
        self.s_shift, self.s_scale = self.s_shift.to(device), self.s_scale.to(device)
        self.a_shift, self.a_scale = self.a_shift.to(device), self.a_scale.to(device)
        self.out_shift, self.out_scale = self.out_shift.to(device), self.out_scale.to(device)
        # if some state dimensions have very small variations, we will force it to zero
        self.mask = self.out_scale >= 1e-8

        self.transformations = dict(s_shift=self.s_shift, s_scale=self.s_scale,
                                    a_shift=self.a_shift, a_scale=self.a_scale,
                                    out_shift=self.out_shift, out_scale=self.out_scale)

    def forward(self, s, a):
        if s.dim() != a.dim():
            print("State and action inputs should be of the same size")
        # normalize inputs
        s_in = (s - self.s_shift)/(self.s_scale + 1e-8)
        a_in = (a - self.a_shift)/(self.a_scale + 1e-8)
        out = torch.cat([s_in, a_in], -1)
        for i in range(len(self.fc_layers)-1):
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out)
        out = self.fc_layers[-1](out)
        if self._apply_out_transforms:
            out = out * (self.out_scale + 1e-8) + self.out_shift
            out = out * self.mask if self.use_mask else out
            out = out + s if self.residual else out
        return out

    def get_params(self):
        network_weights = [p.data for p in self.parameters()]
        transforms = (self.s_shift, self.s_scale,
                      self.a_shift, self.a_scale,
                      self.out_shift, self.out_scale)
        return dict(weights=network_weights, transforms=transforms)

    def set_params(self, new_params):
        new_weights = new_params['weights']
        s_shift, s_scale, a_shift, a_scale, out_shift, out_scale = new_params['transforms']
        for idx, p in enumerate(self.parameters()):
            p.data = new_weights[idx]
        self.set_transformations(s_shift, s_scale, a_shift, a_scale, out_shift, out_scale)


class RewardNet(nn.Module):
    def __init__(self, state_dim, act_dim, 
                 hidden_size=(64,64),
                 s_shift = None,
                 s_scale = None,
                 a_shift = None,
                 a_scale = None,
                 seed=123,
                 ):
        super(RewardNet, self).__init__()
        torch.manual_seed(seed)
        self.state_dim, self.act_dim, self.hidden_size = state_dim, act_dim, hidden_size
        self.layer_sizes = (state_dim + act_dim + state_dim, ) + hidden_size + (1, )
        # hidden layers
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])
                                        for i in range(len(self.layer_sizes)-1)])
        self.nonlinearity = torch.relu
        self.set_transformations(s_shift, s_scale, a_shift, a_scale)

    def set_transformations(self, s_shift=None, s_scale=None,
                            a_shift=None, a_scale=None,
                            out_shift=None, out_scale=None):

        if s_shift is None:
            self.s_shift, self.s_scale       = torch.zeros(self.state_dim), torch.ones(self.state_dim)
            self.a_shift, self.a_scale       = torch.zeros(self.act_dim), torch.ones(self.act_dim)
            self.sp_shift, self.sp_scale     = torch.zeros(self.state_dim), torch.ones(self.state_dim)
            self.out_shift, self.out_scale   = 0.0, 1.0 
        elif type(s_shift) == torch.Tensor:
            self.s_shift, self.s_scale       = s_shift, s_scale
            self.a_shift, self.a_scale       = a_shift, a_scale
            self.sp_shift, self.sp_scale     = s_shift, s_scale
            self.out_shift, self.out_scale   = out_shift, out_scale
        elif type(s_shift) == np.ndarray:
            self.s_shift, self.s_scale       = torch.from_numpy(s_shift).float(), torch.from_numpy(s_scale).float()
            self.a_shift, self.a_scale       = torch.from_numpy(a_shift).float(), torch.from_numpy(a_scale).float()
            self.sp_shift, self.sp_scale     = torch.from_numpy(s_shift).float(), torch.from_numpy(s_scale).float()
            self.out_shift, self.out_scale   = out_shift, out_scale
        else:
            print("Unknown type for transformations")
            quit()

        device = next(self.parameters()).data.device
        self.s_shift, self.s_scale   = self.s_shift.to(device), self.s_scale.to(device)
        self.a_shift, self.a_scale   = self.a_shift.to(device), self.a_scale.to(device)
        self.sp_shift, self.sp_scale = self.sp_shift.to(device), self.sp_scale.to(device)

        self.transformations = dict(s_shift=self.s_shift, s_scale=self.s_scale,
                                    a_shift=self.a_shift, a_scale=self.a_scale,
                                    out_shift=self.out_shift, out_scale=self.out_scale)

    def forward(self, s, a, sp):
        # The reward will be parameterized as r = f_theta(s, a, s').
        # If sp is unavailable, we can re-use s as sp, i.e. sp \approx s
        if s.dim() != a.dim():
            print("State and action inputs should be of the same size")
        # normalize all the inputs
        s = (s - self.s_shift) / (self.s_scale + 1e-8)
        a = (a - self.a_shift) / (self.a_scale + 1e-8)
        sp = (sp - self.sp_shift) / (self.sp_scale + 1e-8)
        out = torch.cat([s, a, sp], -1)
        for i in range(len(self.fc_layers)-1):
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out)
        out = self.fc_layers[-1](out)
        out = out * (self.out_scale + 1e-8) + self.out_shift
        return out

    def get_params(self):
        network_weights = [p.data for p in self.parameters()]
        transforms = (self.s_shift, self.s_scale,
                      self.a_shift, self.a_scale)
        return dict(weights=network_weights, transforms=transforms)

    def set_params(self, new_params):
        new_weights = new_params['weights']
        s_shift, s_scale, a_shift, a_scale = new_params['transforms']
        for idx, p in enumerate(self.parameters()):
            p.data = new_weights[idx]
        self.set_transformations(s_shift, s_scale, a_shift, a_scale)


def fit_model(nn_model, X, Y, optimizer, loss_func,
              batch_size, epochs, max_steps=1e10):
    """
    :param nn_model:        pytorch model of form Y = f(*X) (class)
    :param X:               tuple of necessary inputs to the function
    :param Y:               desired output from the function (tensor)
    :param optimizer:       optimizer to use
    :param loss_func:       loss criterion
    :param batch_size:      mini-batch size
    :param epochs:          number of epochs
    :return:
    """

    assert type(X) == tuple
    for d in X: assert type(d) == torch.Tensor
    assert type(Y) == torch.Tensor
    device = Y.device
    for d in X: assert d.device == device

    num_samples = Y.shape[0]
    epoch_losses = []
    steps_so_far = 0
    for ep in range(epochs):
        rand_idx = torch.LongTensor(np.random.permutation(num_samples)).to(device)
        ep_loss = 0.0
        num_steps = int(num_samples // batch_size)
        for mb in range(num_steps):
            data_idx = rand_idx[mb*batch_size:(mb+1)*batch_size]
            batch_X  = [d[data_idx] for d in X]
            batch_Y  = Y[data_idx]
            optimizer.zero_grad()
            Y_hat    = nn_model.forward(*batch_X)
            loss = loss_func(Y_hat, batch_Y)
            loss.backward()
            optimizer.step()
            ep_loss += loss.to('cpu').data.numpy()
        print(ep_loss * 1.0/num_steps)
        epoch_losses.append(ep_loss * 1.0/num_steps)
        steps_so_far += num_steps
        if steps_so_far >= max_steps:
            print("Number of grad steps exceeded threshold. Terminating early..")
            break
    return epoch_losses
