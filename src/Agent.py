import os
import numpy as np
np.random.seed(0)
import random
random.seed(0)
import copy
import torch
torch.manual_seed(0)
from collections import deque
import time


### User specified parameters ###
USE_BIAS = False
INIT_MEAN = 0.0  # mean of initial training parameters
INIT_STD = 0.05  # standard deviation of initial training parameters

# CAPACITY = 1E3
# BATCH_SIZE = 32
# GAMMA = 0.99
# TARGET_UPDATE_FREQ = 100
#################################


class NN(torch.nn.Module):
    def __init__(self, n_node_inputs: int, n_edge_inputs: int, n_feature_outputs: int, n_action_types: int, 
                 use_gpu: bool, batch_size: int):
        super(NN, self).__init__()
        self.l1_1 = torch.nn.Linear(n_edge_inputs, n_feature_outputs, bias=USE_BIAS)
        self.l1_2 = torch.nn.Linear(n_node_inputs, n_feature_outputs, bias=USE_BIAS)
        self.l1_3 = torch.nn.Linear(n_feature_outputs, n_feature_outputs, bias=USE_BIAS)
        # self.l1_4 = torch.nn.Linear(n_feature_outputs,n_feature_outputs,bias=USE_BIAS)
        # self.l1_5 = torch.nn.Linear(n_feature_outputs,n_feature_outputs,bias=USE_BIAS)

        self.l2_1 = torch.nn.Linear(n_feature_outputs*2, n_action_types, bias=USE_BIAS)
        # self.l2_2 = torch.nn.Linear(n_feature_outputs,n_feature_outputs,bias=USE_BIAS)
        # self.l2_3 = torch.nn.Linear(n_feature_outputs,n_feature_outputs,bias=USE_BIAS)

        self.ActivationF = torch.nn.LeakyReLU(0.2)

        self.Initialize_weight()

        self.n_feature_outputs = n_feature_outputs
        self.BATCH_SIZE = batch_size

        if use_gpu:
            self.to('cuda')
            self.device = torch.device('cuda')
        else:
            self.to('cpu')
            self.device = torch.device('cpu')

    def Connectivity(self, connectivity, n_nodes):
        '''
        connectivity[n_edges, 2]
        '''
        n_edges = connectivity.shape[0]
        order = np.arange(n_edges)
        adjacency = torch.zeros(n_nodes, n_nodes, dtype=torch.float32, device=self.device, requires_grad=False)
        incidence = torch.zeros(n_nodes, n_edges, dtype=torch.float32, device=self.device, requires_grad=False)

        for i in range(2):
            adjacency[connectivity[:,i], connectivity[:,(i+1)%2]] = 1
        incidence[connectivity[:,0], order] = -1
        incidence[connectivity[:,1], order] = 1

        incidence_A = torch.abs(incidence)#.to_sparse()
        incidence_1 = (incidence == -1).type(torch.float32)
        incidence_2 = (incidence == 1).type(torch.float32)

        return incidence_A, incidence_1, incidence_2, adjacency

    def Initialize_weight(self):
        for m in self._modules.values():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=INIT_MEAN, std=INIT_STD)

    def mu(self, v, mu, w, incidence_A, incidence_1, incidence_2, adjacency, mu_iter):
        '''
        - v (array[n_nodes, n_node_features])
        - mu(array[n_edges, n_edge_out_features])
        - w (array[n_edges, n_edge_in_features])
        '''

        h1 = self.ActivationF(self.l1_1.forward(w))
        h2_0 = self.ActivationF(self.l1_2.forward(v))
        h2 = torch.mm(incidence_A.T, h2_0)  # 矩陣相乘
        if mu_iter == 0:
            mu = h1 + h2
        else:
            h3_0 = torch.mm(incidence_A, mu)
            n_connect_edges_1 = torch.sum(torch.mm(adjacency,incidence_1),axis=0).repeat(self.n_feature_outputs,1).T
            n_connect_edges_2 = torch.sum(torch.mm(adjacency,incidence_2),axis=0).repeat(self.n_feature_outputs,1).T
            h3_1 = self.ActivationF(self.l1_3.forward((torch.mm(incidence_1.T,h3_0)-mu)/n_connect_edges_1))
            h3_2 = self.ActivationF(self.l1_3.forward((torch.mm(incidence_2.T,h3_0)-mu)/n_connect_edges_2))
            mu = h1 + h2 + h3_1 + h3_2
        return mu
        
    def Q(self, mu, n_edges):
        if type(n_edges) is int: # normal operation
            mu_sum = torch.sum(mu, axis=0)
            mu_sum = mu_sum.repeat(n_edges, 1)
        else: # for mini-batch training
            mu_sum = torch.zeros((n_edges[-1],self.n_feature_outputs), dtype=torch.float32, device=self.device)
            for i in range(self.BATCH_SIZE):
                mu_sum[n_edges[i]:n_edges[i+1],:] = torch.sum(mu[n_edges[i]:n_edges[i+1],:], axis=0)

        Q = self.l2_1(torch.cat((mu_sum,mu),1))  # shape: [total n_edges, n_action_types=2]
        
        return Q

    def Forward(self, v, w, connectivity, n_mu_iter=3, nm_batch=None):
        '''
        - v [n_nodes, n_node_in_features]
        - w [n_edges, n_edge_in_features]
        - connectivity [n_edges, 2]
        - nm_batch [BATCH_SIZE] : int
        '''
        IA, I1, I2, D = self.Connectivity(connectivity, v.shape[0])

        if type(v) is np.ndarray: 
            v = torch.tensor(v, dtype=torch.float32, device=self.device, requires_grad=False)
        if type(w) is np.ndarray:
            w = torch.tensor(w, dtype=torch.float32, device=self.device, requires_grad=False)
        mu = torch.zeros((connectivity.shape[0],self.n_feature_outputs), device=self.device)

        for i in range(n_mu_iter):
            mu = self.mu(v, mu, w, IA, I1, I2, D, mu_iter=i)  # shape: [total n_edges, n_feature_outputs]
            # print("iter {0}: {1}".format(i,mu.norm(p=2)))

        if nm_batch is None:
            Q = self.Q(mu, w.shape[0])
        else:
            Q = self.Q(mu, nm_batch)

        return Q

    def Save(self, filename, directory=""):
        torch.save(self.to('cpu').state_dict(), os.path.join(directory, filename))
    
    def Load(self, filename, directory=""):
        self.load_state_dict(torch.load(os.path.join(directory, filename)))


class Brain():
    def __init__(self, n_node_inputs: int, n_edge_inputs: int, n_feature_outputs: int, n_action_types: int, 
                 use_gpu: bool, loss_type: str, capacity: int, batch_size: int, gamma: float, optimizer: str):
        if use_gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.loss_type = loss_type
        self.CAPACITY = capacity
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma

        self.model = NN(n_node_inputs, n_edge_inputs, n_feature_outputs, n_action_types, use_gpu, batch_size)
        self.target_model = NN(n_node_inputs, n_edge_inputs, n_feature_outputs, n_action_types, use_gpu, batch_size)
        self.memory = deque()
        self.lossfunc = torch.nn.MSELoss()
        if optimizer == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=1.0e-5)
        elif optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0e-5)

    def store_experience(self, connectivity, v, w, action, reward, v_next, w_next, ep_end, infeasible_actions):
        v = torch.tensor(v, dtype=torch.float32, device=self.device, requires_grad=False)
        w = torch.tensor(w, dtype=torch.float32, device=self.device, requires_grad=False)
        v_next = torch.tensor(v_next, dtype=torch.float32, device=self.device, requires_grad=False)
        w_next = torch.tensor(w_next, dtype=torch.float32, device=self.device, requires_grad=False)
        self.memory.append((connectivity, v, w, action, reward, v_next, w_next, ep_end, infeasible_actions))
        if len(self.memory) > self.CAPACITY:
            self.memory.popleft()

    def sample_batch(self):
        batch = random.sample(self.memory, self.BATCH_SIZE)

        c_batch = np.zeros((0,2), dtype=int)
        v_batch = torch.cat([batch[i][1] for i in range(self.BATCH_SIZE)], dim=0)
        w_batch = torch.cat([batch[i][2] for i in range(self.BATCH_SIZE)], dim=0)
        a_batch = np.vstack([batch[i][3] for i in range(self.BATCH_SIZE)])
        r_batch = torch.tensor([batch[i][4] for i in range(self.BATCH_SIZE)], dtype=torch.float32, device=self.device, requires_grad=False)
        v2_batch = torch.cat([batch[i][5] for i in range(self.BATCH_SIZE)], dim=0)
        w2_batch = torch.cat([batch[i][6] for i in range(self.BATCH_SIZE)], dim=0)
        ep_end_batch = torch.tensor([batch[i][7] for i in range(self.BATCH_SIZE)], dtype=bool, device=self.device, requires_grad=False)
        infeasible_a_batch = np.concatenate([batch[i][8] for i in range(self.BATCH_SIZE)], axis=0)
        nm_batch = np.zeros(self.BATCH_SIZE+1, dtype=int)

        nn = 0  # number of nodes
        nm = 0  # number of mu iteraions
        for i in range(self.BATCH_SIZE):
            c_batch = np.concatenate((c_batch,batch[i][0]+nn), axis=0)
            nn += batch[i][1].shape[0]
            nm += batch[i][2].shape[0]
            nm_batch[i+1] = nm
            
        a_batch[:,0] += nm_batch[:-1]

        return c_batch, v_batch, w_batch, a_batch, r_batch, v2_batch, w2_batch, ep_end_batch, infeasible_a_batch, nm_batch

    def experience_replay(self):
        if len(self.memory) < self.BATCH_SIZE:
            return float('nan')
        
        c, v, w, a, r, v_next, w_next, ep_end, infeasible_actions, nm = self.sample_batch()
        self.optimizer.zero_grad()
        if self.loss_type == "Japan":
            loss = self.calc_loss_Japan(c, v, w, a, r, v_next, w_next, ep_end, infeasible_actions, nm)
        elif self.loss_type == "Taiwan":
            loss = self.calc_loss_Taiwan(c, v, w, a, r, v_next, w_next, ep_end, infeasible_actions, nm)
        loss.backward()
        self.optimizer.step()
        return loss.detach().to('cpu').numpy()

    def calc_loss_Japan(self, c, v, w, action, r, v_next, w_next, ep_end, infeasible_actions, nm):
        # action Q-network
        Q_value = self.model.Forward(v, w, c, nm_batch=nm)
        Q_current = Q_value[action[:,0], action[:,1]]
        
        # value Q-network
        tmp = self.target_model.Forward(v_next, w_next, c, nm_batch=nm).detach()
        tmp[infeasible_actions] = -1.0e20
        Q_max_next = torch.tensor([tmp[nm[i]:nm[i+1]].max() for i in range(self.BATCH_SIZE)], dtype=torch.float32, device=self.device, requires_grad=False)
        Q_target = r + (self.GAMMA * Q_max_next) * ~ep_end
        
        loss = self.lossfunc(Q_current, Q_target)
        return loss
    
    def calc_loss_Taiwan(self, c, v, w, action, r, v_next, w_next, ep_end, infeasible_actions, nm):
        # action Q-network
        Q_value = self.model.Forward(v, w, c, nm_batch=nm)
        Q_current = Q_value[action[:,0], action[:,1]]

        Q_value_next = self.model.Forward(v_next, w_next, c, nm_batch=nm).detach()
        action_max_next = torch.tensor([(Q_value_next[nm[i]:nm[i+1]].argmax(dim=0) + nm[i]).tolist() for i in range(self.BATCH_SIZE)], dtype=torch.int64, device=self.device, requires_grad=False)

        # value Q-network
        tmp = self.target_model.Forward(v_next, w_next, c, nm_batch=nm).detach()
        action_type = action[0,1]  # same action type for all actions in the batch, dec: 0, inc: 1
        Q_value_max_next = tmp[action_max_next[:,action_type], action[:,1]]
        Q_target = r + (self.GAMMA * Q_value_max_next) * ~ep_end

        loss = self.lossfunc(Q_current, Q_target)
        return loss

    def decide_action(self, v, w, c, eps, infeasible_actions):
        Q = self.model.Forward(v, w, c).detach().to('cpu').numpy()
        
        if np.random.rand() > eps:  # high probability --> greedy
            a_flatten = np.ma.masked_where(infeasible_actions, Q).argmax()  # mask elements where condition is True, and return the index of the maximum value for flattened array
            a = np.divmod(a_flatten, Q.shape[1])
        else:                       # low probability --> random
            feasible_a_indices = np.argwhere(~infeasible_actions)  # returns the indices of all non-zero elements
            action, action_type = np.asarray(feasible_a_indices[np.random.randint(np.shape(feasible_a_indices)[0])])
            a = action, action_type  # action_type = 0: dec, 1: inc

        return a, Q[a]
    
    def gnn_coupling(self):
        self.target_model.l1_1.weight.data = copy.deepcopy(self.model.l1_1.weight.data)
        self.target_model.l1_2.weight.data = copy.deepcopy(self.model.l1_2.weight.data)
        self.target_model.l1_3.weight.data = copy.deepcopy(self.model.l1_3.weight.data)
        if USE_BIAS:
            self.target_model.l1_1.bias.data = copy.deepcopy(self.model.l1_1.bias.data)
            self.target_model.l1_2.bias.data = copy.deepcopy(self.model.l1_2.bias.data)
            self.target_model.l1_3.bias.data = copy.deepcopy(self.model.l1_3.bias.data)
            

class Agent():
    def __init__(self, n_node_inputs: int, n_edge_inputs: int, n_feature_outputs: int, n_action_types: int, 
                 use_gpu: bool, loss_type: str, capacity: int, batch_size: int, gamma: float, target_update_freq: int, optimizer: str):
        self.brain = Brain(n_node_inputs, n_edge_inputs, n_feature_outputs, n_action_types, use_gpu, loss_type, capacity, batch_size, gamma, optimizer)
        self.step = 0
        self.target_update_freq = target_update_freq

    def update_q_function(self, smooth_update=False):
        loss = self.brain.experience_replay()
        if self.step % self.target_update_freq == 0:
            self.brain.target_model = copy.deepcopy(self.brain.model)
            # _ = self.brain.target_model.load_state_dict(self.brain.model.state_dict())  # has no impact on the training result
        self.step += 1
        return loss
        
    def get_action(self, v, w, c, eps, infeasible_actions):
        action, q = self.brain.decide_action(v, w, c, eps=eps, infeasible_actions=infeasible_actions)
        return action, q
    
    def memorize(self, c, v, w, action, reward, v_next, w_next, ep_end, infeasible_actions):
        self.brain.store_experience(c, v, w, action, reward, v_next, w_next, ep_end, infeasible_actions)
