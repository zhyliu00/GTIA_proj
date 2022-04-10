import torchvision
import sys
import subprocess
from model import  GCN,dqn_agent
from env import DRLA_env
from torchsummary import summary
import networkx as nx
import copy
import numpy as np
import pickle
import json
import sklearn
import torch.nn as nn
import torch
import dgl
import argparse
import datetime
import os
import time
import copy
curr_time = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")
# print(os.path.abspath(__file__))
curr_path = os.path.dirname(os.path.abspath(__file__))
curr_path = os.path.dirname(__file__)
parser = argparse.ArgumentParser()

parser.add_argument('--batch_size',type=int,default=128)
parser.add_argument('--gpu_id', type=int,default=0)
parser.add_argument('--epochs', type=int,default=50)
parser.add_argument('--hidden', type=int,default=128)
parser.add_argument('--lr',type=float,default=1e-3)
parser.add_argument('--test_folder', type = str, default = None)
parser.add_argument('--phase', type = str,default='test')
parser.add_argument('--g_hidden',type=int,default=64)
parser.add_argument('--J',type=int,default=8)
parser.add_argument('--K',type=int,default=8)
parser.add_argument('--N',type=int,default=100)
parser.add_argument('--layers',type=int,default=2)
args = parser.parse_args()

config = {
    # 'device': torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu"),
    'device' : torch.device('cpu'),
    'batch' : args.batch_size,
    'lr' : args.lr,
    'max_epohs' : args.epochs,
    'save_every' :10,     
    'double_dqn' : True,
    'max_episodes' : 2000,
    'argument':vars(args),
    'eval_eps' : 30, # number of episodes for evaluating
    'max_steps' : 200,
    'target_update' : 4,
    'algo' : "DQN",  # name of algo
    'gamma' : 0.90,
    'epsilon_start' : 0.90,  # start epsilon of e-greedy policy
    'epsilon_end' : 0.05,
    'epsilon_decay' : 3000,
    'memory_capacity' : 10000 , # capacity of Replay Memory
    'batch_size' : args.batch_size,
    
    'g_hidden':args.g_hidden,
    'J': args.J,
    'K': args.K,
    'N': args.N,
    'layers': args.layers,
    'delta_g' : 10,
    'delta_l' : 5,
    'total_worker': args.N,
    'total_episode': 10000,
    
}
if(args.test_folder != None):
    curr_time = args.test_folder
config['result_path']= os.path.join(curr_path , "outputs/DRLA/" + curr_time + '/results/')  # path to save results
config['model_path']= os.path.join( curr_path ,"outputs/DRLA/" + curr_time + '/models/')  # path to save models

def check_path(path):
    if(not os.path.exists(path)):
        os.makedirs(path)

def train_DRLA():
    
    env = DRLA_env(config)

    N = config['N']
    agent = dqn_agent(N,config)
    check_path(config['result_path'])
    check_path(config['model_path'])
    
    #######
    # 问题：sigma g选大了，welfare都是负数
    rewards = []
    welfares = []
    best_welfare = -100
    result = {
        'ep_rwd': []
    }
    for epoch in range(config['total_episode']):
        state = env.reset()
        done = False
        ep_reward = 0.0
        cur_reward = []
        # print(torch.sum(env.states[:,2 * args.g_hidden]))
        while(not done):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            # print(reward)
            agent.memory.push(state, action, reward ,next_state, done)
            state = next_state
            agent.update()
            ep_reward += reward
            if(done):
                welfare, workerList = env.finalize()
                
                if(welfare > best_welfare):
                    best_welfare = welfare
                    dqn_agent.save(config['model_path'])
                
            cur_reward.append(reward)
            
        print(cur_reward)
        print('finish. welfare : {}. workerList : {}\n------------'.format(welfare, workerList))
        rewards.append(ep_reward)
        welfares.append(welfare)
        result['ep_rwd'].append(float(welfare.detach().cpu().numpy()))
    # print('rewards : {}'.format(rewards))
        json.dump(result, open('./result.json','w'), indent=2)
    print('welfares : {}'.format(welfares))
    

if __name__ == "__main__":
    
    ## Data input
    train_DRLA()
    
    # # args.test_folder != None means test mode
    # if(args.phase == 'test'):
    #     for k , model in Models.items():
    #         state = torch.load(config['model_path'] + '{}.pth'.format(k),map_location=config['device'])
    #         model.load_state_dict(state)
    
    # # not a end to end task
    
    # if(args.phase == 'train'):
    #     loss_arr = train_vlocal(Models,env)
    
    # test_vlocal(Models,env)
    
    # if(args.phase == 'train'):
    #     with open('./cur_train_folder','w') as f:
    #         f.write(curr_time)
        