from ast import Param
import numpy as np
import pickle
import json
import sklearn
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch
import dgl
import dgl.nn as dnn
import os
import math
from torch.nn.parameter import Parameter, UninitializedParameter
import random
from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair, check_eq_shape, dgl_warning

class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, layers, residual = 0,):
        super(GCN, self).__init__()
        self.gconv = nn.ModuleList([])
        for i in range(layers):
            self.gconv.append(dnn.GraphConv(in_feats = input_size, out_feats = hidden_size, activation = F.relu))

        self.residual = residual
        # self.output = OutputLayer(hidden_size,his_length + 1 - 2**(control_str.count('T')))
        # self.output = FullyConvLayer(hidden_size)
        
    def forward(self,graph,x,e_weight):
        # x : [N, F]
        h = x
        for idx, layer in enumerate(self.gconv):
            h_nxt = layer(graph,h,edge_weight = e_weight)                
            # [N, F']
            if(h_nxt.shape[-1] == h.shape[-1] and self.residual):
                h_nxt = h_nxt + h
            h = h_nxt
        return h

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        # state : [N, F]
        # action : 1
        # reward : 1
        # next_state : [N, F]
        # done : 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        if(not isinstance(state,torch.Tensor)):
            state = torch.tensor(state)
        
        if(not isinstance(action,torch.Tensor)):
            action = torch.tensor(action)
            
        if(not isinstance(reward,torch.Tensor)):
            reward = torch.tensor(reward)
            
        if(not isinstance(next_state,torch.Tensor)):
            next_state = torch.tensor(next_state)
            
        if(not isinstance(done,torch.Tensor)):
            done = torch.tensor(done)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        if(isinstance(state[0],torch.Tensor)):
            state = torch.stack(list(state),dim=0)
        if(isinstance(action[0],torch.Tensor)):
            action = torch.stack(list(action),dim=0)
        if(isinstance(reward[0],torch.Tensor)):
            reward = torch.stack(list(reward),dim=0)
        if(isinstance(next_state[0],torch.Tensor)):
            next_state = torch.stack(list(next_state),dim=0)
        if(isinstance(done[0],torch.Tensor)):
            done = torch.stack(list(done),dim=0)
            
        # return all torch.tensor
        # [B, its shape]
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class dqn_model(nn.Module):
    def __init__(self,g_hidden,J,K,N):
        super(dqn_model,self).__init__()
        self.g_hidden = g_hidden
        self.J = J
        self.K = K
        self.N = N
        self.mat_params = nn.ParameterDict({
            '1':Parameter(torch.empty(2 * g_hidden + 3,2 * g_hidden + 3)),
            '2':Parameter(torch.empty(2 * g_hidden + 3,1)),
            # '3':Parameter(torch.empty((2 * g_hidden + 3, 1))),
            '3':Parameter(torch.empty((1, 1))),
            '4':Parameter(torch.empty((1, 1))),
            '51':Parameter(torch.empty((J,K,2,K))),
            '61':Parameter(torch.empty((J,K,N,K))),
            '52':Parameter(torch.empty((J,K,K,1))),
            '62':Parameter(torch.empty((J,K,N,1)))
        })
        # print(self.N,N)
        self._reset_params()
    def _reset_params(self):
        for k,v in self.mat_params.items():
            init.kaiming_uniform_(v, a=math.sqrt(5))
        
    def unpack(self,states):
        # print(states.shape)
        # states : [B, N ,F]
        g_hidden = self.g_hidden
        v = states[: ,:, : 2 * g_hidden]
        s = states[: ,:, 2*g_hidden].unsqueeze(-1)
        C = states[: ,:, 2*g_hidden + 1].unsqueeze(-1)
        h = states[: ,:, 2*g_hidden + 2].unsqueeze(-1)
        b = states[: ,:, 2*g_hidden + 3].unsqueeze(-1)
        d = states[: ,:, 2*g_hidden + 4].unsqueeze(-1)
        sigma = states[:, :, 2*g_hidden + 5].unsqueeze(-1)
        return v,s,C,h,b,d,sigma
    def forward(self, states):
        # v : [B, N, 2w]
        # else: [B, N, 1]
        v,s,C,h,b,d,sigma = self.unpack(states)
        feat_1 = torch.cat([v,s,C,h],dim = 2)
        # print(v.shape, s.shape, C.shape, feat_1.shape)
        # feat_1 : [B, N , 2w + 3]
        
        Q_1 = F.linear(feat_1, self.mat_params['1'].T)
        Q_1 = F.relu(Q_1)
        # print(Q_1.shape, self.mat_params['2'].shape)
        Q_1 = F.linear(Q_1, self.mat_params['2'].T)
        # Q_1 : [B, N ,1]
        
        
        # b : [B, N, 1]
        
        Q_2 = F.linear(-b, torch.exp(self.mat_params['3']).T)
        # here problem
        
        feat_2 = torch.cat([d,-sigma],dim = 2)
        
        # [B, N, 2] [J, K, 2, K] -> [B, J, K, N, K]
        
        g_1 = torch.einsum('bni,jkid->bjknd',[feat_2, torch.exp(self.mat_params['51'])])
        # [B, J, K, N, K] + [J, K, N, K]
        # print(g_1.shape, self.mat_params['61'].shape)
        g_1 = g_1 + self.mat_params['61']
        g_1 = F.relu(g_1)
        
        # [B, J, K, N, K] [J, K, K ,1] -> [B, J, K, N, 1]
        g_1 = torch.einsum('bjkni,jkid->bjknd',[g_1, torch.exp(self.mat_params['52'])])
        
        # boradcast [B, J, K, N, 1] [J, K, N, 1]
        g_1 = g_1 + self.mat_params['62']
        g_1 = F.relu(g_1)
        
        
        g_1.squeeze_(-1)
        
        # [B, J, K, N]
        # find the max-min
        
        g_1 = torch.min(g_1,dim=2,keepdim=False)[0]
        # [B,J,N]
        g_1 = torch.max(g_1,dim=1,keepdim=False)[0]
        # [B,N]
        g_1.unsqueeze_(-1)
        # [B,N,1]
        # print(g_1.shape, self.mat_params['4'].shape)
        Q_3 = F.linear(g_1, torch.exp(self.mat_params['4']).T)
        
        # print("shape Q_1 : {}, Q_2 : {}, Q_3 : {}".format(Q_1.shape,Q_2.shape,Q_3.shape))
        Q = Q_1 + Q_2 + Q_3
        # [N, 1]
        return Q
                
                
class dqn_agent:
    def __init__(self,action_dim,cfg):
        # target_net 是用来固定算分的
        # policy_net 是用来更新参数的
        self.g_hidden = cfg['g_hidden']
        self.J = cfg['J']
        self.K = cfg['K']
        self.N = cfg['N']
        self.action_dim = action_dim
        self.cfg = cfg
        self.target_net = dqn_model(self.g_hidden,self.J,self.K,self.N).to(cfg['device'])
        self.policy_net = dqn_model(self.g_hidden,self.J,self.K,self.N).to(cfg['device'])
        
        self.action_space = action_dim
        self.memory_cap = 1e6
        self.memory = ReplayBuffer(self.memory_cap)
        self.batch_size = cfg['batch_size']
        self.epsilon = lambda frame_idx: cfg['epsilon_end'] + \
            (cfg['epsilon_start'] - cfg['epsilon_end']) * \
            math.exp(-1. * frame_idx / cfg['epsilon_decay'])
        self.frame_idx = 0
        self.cfg = cfg

        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),lr = cfg['lr'])
        self.double_dqn = cfg['double_dqn']
    def select_action(self,state):
        # double dqn, 使用更新后的net去找动作
        # vanilla dqn, 使用更新前的net去找动作
        # [N, F], no batch
        double_dqn = self.double_dqn
        used_policy = None
        
        if(double_dqn):
            used_policy = self.policy_net
        else:
            used_policy = self.target_net
        with torch.no_grad():
            
            state = torch.tensor(state, device=self.cfg['device'], dtype=torch.float32).unsqueeze(0)
            # state [B, N, F]
            
            q_values = used_policy(state)
            # q_values [B, N, 1]
            print('during select action, q_values.shape is {}'.format(q_values.shape))
            
        s = state[0, :, 2*self.g_hidden].unsqueeze(-1)
        # [N]
        self.frame_idx+=1
        epsilon = self.epsilon(self.frame_idx)
        
        if(np.random.random_sample() > epsilon):
            q_values = q_values.squeeze(0)
            q_values = q_values.squeeze(-1)
            # [N]
            sorted, indices = torch.sort(q_values)
            for i in indices:
                if(s[i]==1):
                    continue
                else:
                    action = i
            
            # action = torch.max(q_values,dim = 1, keepdim=False)[1].cpu().numpy()
            # action = action.squeeze(0)
            # action = int(q_values.max(0)[1].cpu().numpy())
        else:
            action = np.random.randint(self.action_space)
            while(s[action]==1):
                action = np.random.randint(self.action_space)
        return int(action)
        # exp_values = torch.exp(action_values)
        # sum_exp_values = torch.sum(exp_values)
        # prob = exp_values / sum_exp_values

    def predict(self, state):
        # [N, F]
        # No batch
        with torch.no_grad():
            
            state = torch.tensor([state], device=self.cfg['device'], dtype=torch.float32)
            s = state[0, :, 2*self.g_hidden].unsqueeze(-1)
            q_values = self.policy_net(state)
            q_values = q_values.squeeze(0)
            q_values = q_values.squeeze(-1)
            # [N]
            sorted, indices = torch.sort(q_values)
            for i in indices:
                if(s[i]==1):
                    continue
                else:
                    action = i
            # action = torch.max(q_values,dim = 1, keepdim=False)[1].cpu().numpy()
            # action = action.squeeze(0)
        return int(action)
    def update(self):
        if len(self.memory) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)

        # action_batch , reward_batch, done_batch: [B, 1]
        # state_batch : [B, N, F]
        
        # state_batch = torch.tensor(
        #     state_batch, device=self.cfg.device, dtype=torch.float).unsqueeze(1)
        # action_batch = torch.tensor(action_batch, device=self.cfg.device)
        # action_batch.unsqueeze_(1)
        # reward_batch = torch.tensor(
        #     reward_batch, device=self.cfg.device, dtype=torch.float)  # tensor([1., 1.,...,1])
        # next_state_batch = torch.tensor(
        #     next_state_batch, device=self.cfg.device, dtype=torch.float).unsqueeze(1)
        # done_batch = torch.tensor(np.float32(
        #     done_batch), device=self.cfg.device)
        # next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        
        
        next_q_values = self.target_net(next_state_batch).detach()
        # [B, N, 1]
        if (self.double_dqn):
            next_q_policy = self.policy_net(next_state_batch)
            # [B, N, 1]
            next_action = torch.max(next_q_policy, dim=1, keepdim=False)[1]
            next_action.unsqueeze(1)
            # [B, 1, 1]
            next_q_values = next_q_values.gather(1, next_action).squeeze(1)
            # [B, 1]
        else:
            # [B, N, 1]
            next_q_values = next_q_values.max(1)[0]
            # [B, 1]
            
            
            
        q_values = self.policy_net(state_batch)
        #[B, N, 1]
        action_batch.unsqueeze_(1)
        # action_batch [B, 1, 1]
        q_values = q_values.gather(1, action_batch)
        #[B, 1, 1]
        q_values = q_values.squeeze(1)
        #[B, 1]
        
        target = reward_batch + self.cfg['gamma'] * next_q_values * (1 - done_batch)
        # target.unsqueeze_(1)
        
        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




    def save(self, path):
        if(not os.path.exists(path)):
            os.mkdir(path)
        torch.save(self.target_net.state_dict(), os.path.join(path,'dqn_checkpoint.pth'))
        print("Model Saved!")
    def load(self, path):
        self.target_net.load_state_dict(torch.load(os.path.join(path,'dqn_checkpoint.pth')))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)
        print("Model Loaded!")
        
        
