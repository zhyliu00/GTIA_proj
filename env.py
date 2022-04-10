from ast import Global
import copy
import numpy as np
from model import  GCN,dqn_agent
import torch
import torch.nn as nn
import dgl
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import random 
from torch.utils.data import DataLoader
class MNIST_model(nn.Module):
    def __init__(self):
        super(MNIST_model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16 ,5 , 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),
        )
        self.out = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
        
    def forward(self,x):
        h = self.conv(x)
        h = h.view(h.shape[0], -1)
        h = self.out(h)
        # h = F.LogSoftmax(h,dim=1)
        return h
        
    
def Get_MNIST():
    train_mnist =  torchvision.datasets.MNIST(root='./',train=True,download=True,transform=transforms.ToTensor())
    test_mnist =   torchvision.datasets.MNIST(root='./',train=False,download=True,transform=transforms.ToTensor())
    
    return train_mnist, test_mnist

class worker:
    def __init__(self, MNIST_data):
        
        self.gamma = torch.FloatTensor(1).uniform_(1e-5,1e-4)
        self.alpha = torch.FloatTensor(1).uniform_(1e-5,1e-4)
        self.beta = torch.FloatTensor(1).uniform_(1e-2,0.1)
        self.h = torch.FloatTensor(1).uniform_(1e6,1e7)
        self.b = torch.FloatTensor(1).uniform_(0,10)
        
        self.model = MNIST_model()

        self.loss_fn = nn.CrossEntropyLoss()   
        self.opt = torch.optim.Adam(self.model.parameters(), lr = 0.001)
        
        
        
        self.C_total = list(range(1,101))
        self.C = random.randint(2,6)
        self.C_list = random.sample(self.C_total,self.C)
        self.C_total = torch.tensor(self.C_total)
        self.C = torch.tensor([self.C])
        self.C_list = torch.tensor(self.C_list)
        
        self.d_total = list(range(0,60000))
        self.d = random.randint(2,10)
        self.d_list = random.sample(self.d_total,self.d)
        self.d_total = torch.tensor(self.d_total)
        self.d = torch.tensor([self.d])
        self.d_list = torch.tensor(self.d_list)
        
        self.MNIST_data = MNIST_data
        train,test = MNIST_data
        self.train_list = []
        label_cnt = [0] * 10
        for idx in self.d_list:
            self.train_list.append(train[idx])
            label_cnt[train[idx][1]]+=1
        label_cnt = torch.tensor(label_cnt,dtype=torch.float32)
        label_cnt /= len(self.d_list)
        
        self.gnd_distb = [0.0987, 0.1124, 0.0993, 0.1022, 0.0974, 0.0903, 0.0986, 0.1044, 0.0975, 0.0992]
        self.sigma = 0.0
        for idx,v in enumerate(label_cnt):
            self.sigma += torch.abs(label_cnt[idx]-self.gnd_distb[idx])
        self.sigma = torch.clip(self.sigma,max=1.2)
        
        # todo, convert the train to tensor
        self.train_X = None
        self.train_y = None
        for (X,y) in self.train_list:
            X = X.unsqueeze(0)
            y = torch.tensor([y])
            if(self.train_X ==None):
                self.train_X = X
                self.train_y = y
            else:
                self.train_X = torch.cat([self.train_X,X],dim = 0)
                self.train_y = torch.cat([self.train_y,y],dim = 0)

        self.state_feature = torch.tensor([self.C,self.h,self.b,self.d,self.sigma]).unsqueeze(0)
        self.delta_l = 5
    def train(self):
        for epoch in range(self.delta_l):
            pred_y = self.model(self.train_X)
            loss = self.loss_fn(pred_y, self.train_y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        
        return self.model
    

class DRLA_env:
    def __init__(self,cfg):
        self.N = cfg['N']
        self.cfg = cfg
        self.g_hidden = cfg['g_hidden']
        self.delta_l = 5
        self.delta_g = cfg['delta_g']
        self.stp = 0
        self.M = 0.5
        self.R = 30 * 1e3
        self.B = 20 * 1e6
        
        self.MNIST_data = Get_MNIST()
        train, test = self.MNIST_data
        self.test_dataloader = DataLoader(test, batch_size=4096, shuffle=True)
        
        
        N = cfg['N']
        
        ####
        # Initial workers
        self.workers = []
        for i in range(N):
            self.workers.append(worker(self.MNIST_data))

        u = []
        v = []
        ####
        # Construct Graph
        for i in range(N):
            for j in range(i,N):
                C_1 = self.workers[i].C_list
                C_2 = self.workers[j].C_list
                cc = np.intersect1d(C_1,C_2)
                if(len(cc) == 0):
                    u.append(i)
                    u.append(j)
                    v.append(j)
                    v.append(i)
        u = torch.tensor(u)
        v = torch.tensor(v)
        edges = u,v
        self.g = dgl.graph(edges) 
        print('Graph constructed')
        print(self.g)
        
        
        # here self loop weight is 0
        self.g = dgl.add_self_loop(self.g)
        features = torch.ones((N,cfg['g_hidden']))
        
        
        self.g.ndata['f'] = features
        self.Model  = GCN(input_size = cfg['g_hidden'],hidden_size = cfg['g_hidden'],layers = cfg['layers'])
        self.opt = torch.optim.Adam(self.Model.parameters(), lr = cfg['lr'])
        ####
        # Conv
        v_hat = self.Model(self.g,self.g.ndata['f'])
        # v_hat : [N, w]
        v_G = torch.sum(v_hat,dim = 0,keepdim=False)
        v_G = v_G.repeat(N,1)
        print("v_G.shape : {}".format(v_G.shape))
        
        
        s = torch.zeros((N,1))
        for i in range(N):
            self.workers[i].v = v_hat[i].detach()
            self.workers[i].v_G = v_G[i].detach()
            
        
        ####
        # init features
        # [N, 5]
        init_features = None
        for i in range(N):
            if(init_features == None):
                init_features = self.workers[i].state_feature
            else:
                init_features = torch.cat([init_features,self.workers[i].state_feature],dim = 0)
        print('init_features shpe : {}'.format(init_features.shape))
        self.init_features = init_features
        
        init_states = torch.cat([v_hat,v_G,s,self.init_features],dim = 1)
        print('init_states shpe : {}'.format(init_states.shape))
        
        self.states = init_states
        self.Global_model =  MNIST_model()

        ####
        # k
        self.k = [0.0, 0.361, 4.348, 1e-3, 0.993, 0.31, 1.743, 100]
        self.alpha_hat = 5e-2
        self.beta_hat = 5e-5

    def test_MNIST(self):
        loader = self.test_dataloader
        model = self.Global_model
        with torch.no_grad():
            res = 0
            sm = 0
            for idx, (X,y) in enumerate(loader):
                pred_logits = model(X)
                pred_number = torch.max(pred_logits,dim=1)[1]
                
                res += torch.sum(pred_number == y).float()
                sm += len(X)
            res = res / sm
            print('this episode, accuracy is {}'.format(res))
        return res
            

    def finalize(self,phase):
        s = self.states[:,2 * self.g_hidden]
        if(phase == 'test'):
            return self.S(s), torch.nonzero(s), 0.0
        models = []
        for idx, val in enumerate(s):
            if(val):
                models.append(self.workers[idx].train())
        if(len(models)==0):
            return self.S(s), torch.nonzero(s), self.test_MNIST()
        global_model =  self.Global_model
        
        temp_params = {}
        
        for model in models:
            for name, param in model.named_parameters():
                if(name not in temp_params.keys()):
                    temp_params[name] = param.clone()
                else:
                    temp_params[name] += param

        for key,param in global_model.named_parameters():
            temp_params[key] /= len(models)
            param.data.copy_(temp_params[key]) 
            
        for i in range(self.N):
            self.workers[i].model.load_state_dict(self.Global_model.state_dict())
        
        acc = self.test_MNIST()
        # print('returned : {}'.format((self.S(s), torch.nonzero(s), acc)))
        return self.S(s), torch.nonzero(s), acc
        
    def S(self,state):
        selected_worker = []
        for i in range(self.N):
            if(state[i] > 0):
                selected_worker.append(self.workers[i])
        delta = 0.0
        D = 0.0
        sigma_ci = 0.0
        
        platform_cost_ci = 0.0
        if(len(selected_worker) == 0):
            # print('zero')
            return torch.tensor(20)
        
        for worker in selected_worker:
            delta += worker.sigma
            D += worker.d
            ci_1 = worker.d * worker.gamma 
            ci_2 = worker.d * self.delta_l * self.delta_g * self.M * worker.alpha 
            ci_3 = (2**(self.R / (self.B * worker.C )) - 1)* self.B * worker.C * self.M / (worker.h * self.R) * self.delta_g * worker.beta
            ci = ci_1 + ci_2 + ci_3
            
            sigma_ci += ci
            
            
            platform_cost_ci += (2**(self.R / (self.B * worker.C )) - 1)* self.B * worker.C * self.M / (worker.h * self.R) * self.delta_g * self.beta_hat

            
        
        delta /= len(selected_worker)
        
        alpha_delta = self.k[4] * torch.exp(- ((delta + self.k[5] ) / self.k[6]) ** 2)
        
        phi_right = self.k[1] * torch.exp( - self.k[2] * (self.k[3] * D) ** alpha_delta )
        phi = self.k[7] * (alpha_delta - phi_right)
        
        c_hat = self.delta_g * self.M * (len(selected_worker) - 1) * self.alpha_hat + platform_cost_ci
        
        reward = phi - c_hat - sigma_ci
        # print('calc_S. output is : {}'.format(reward))
        # print('phi is {}, c_hat is {}, sigma_ci is {}'.format(phi,c_hat,sigma_ci))
        # print('delta is {}, D is {}, alpha_delta is {}, phi_right is {}'.format(delta,D,alpha_delta,phi_right))
        return reward
        
    def reset(self):
        v_hat = self.Model(self.g,self.g.ndata['f'])
        # v_hat : [N, w]
        v_G = torch.sum(v_hat,dim = 0,keepdim=False)
        v_G = v_G.repeat(self.N,1)
        # print("v_G.shape : {}".format(v_G.shape))
        
        
        s = torch.zeros((self.N,1))
        for i in range(self.N):
            self.workers[i].v = v_hat[i].detach()
            self.workers[i].v_G = v_G[i].detach()
            
        
        init_states = torch.cat([v_hat.detach(),v_G.detach(),s,self.init_features],dim = 1)
        print('init_states shpe : {}'.format(init_states.shape))
        self.states = init_states.to(self.cfg['device'])
        self.stp = 0
        return self.states

    def calc_reward(self,s,a):
        n_s = copy.deepcopy(s.data)
        n_s[a] = 1
        nxt_v = self.S(n_s)
        cur_v = self.S(s)
        reward = nxt_v - cur_v
        # print('nxt_v : {}, cur_v : {}'.format(nxt_v, cur_v))
        return torch.tensor([reward])
    
    def calc_done(self,reward):
        # res = (self.stp > 1 or reward < 0)
        res = self.stp > self.cfg['total_worker'] or reward < 0
        return torch.tensor([res],dtype=torch.int16)
    
    def step(self, action):
        s = self.states[:,2 * self.g_hidden]
        
        
        # self.workers[action].train
        self.stp+=1
        reward = self.calc_reward(s,action)
        if(reward >=0):
            s[action] = 1
        # [1]
        done = self.calc_done(reward)
        # [1]
        
        return self.states, reward, done, {}
    
