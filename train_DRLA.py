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
parser.add_argument('--lr',type=float,default=5e-4)
parser.add_argument('--test_folder', type = str, default = None)
parser.add_argument('--phase', type = str,default='test')
parser.add_argument('--g_hidden',type=int,default=64)
parser.add_argument('--J',type=int,default=8)
parser.add_argument('--K',type=int,default=8)
parser.add_argument('--N',type=int,default=100)
parser.add_argument('--layers',type=int,default=2)
parser.add_argument('--delta_g',type=int,default=10)
parser.add_argument('--delta_l',type=int,default=5)
parser.add_argument('--total_episode',type=int,default=50000)
args = parser.parse_args()

config = {
    'device': torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu"),
    # 'device' : torch.device('cpu'),
    'batch' : args.batch_size,
    'lr' : args.lr,
    'max_epohs' : args.epochs,
    
    'double_dqn' : True,
    'argument':vars(args),
    'target_update' : 4,
    'algo' : "DQN",  # name of algo
    'gamma' : 0.90,
    'epsilon_start' : 0.93,  # start epsilon of e-greedy policy
    'epsilon_end' : 0.07,
    'epsilon_decay' : 3000,
    'memory_capacity' : 10000 , # capacity of Replay Memory
    'batch_size' : args.batch_size,
    
    'g_hidden':args.g_hidden,
    'J': args.J,
    'K': args.K,
    'N': args.N,
    'layers': args.layers,
    'delta_g' : args.delta_g,
    'delta_l' : args.delta_l,
    'total_worker': args.N,
    'total_episode': args.total_episode,
    'test_every' :10
    
}
if(args.test_folder != None):
    curr_time = args.test_folder
config['result_path']= os.path.join(curr_path , "outputs/DRLA/" + curr_time + '/results/')  # path to save results
config['model_path']= os.path.join( curr_path ,"outputs/DRLA/" + curr_time + '/models/')  # path to save models



def check_path(path):
    if(not os.path.exists(path)):
        os.makedirs(path)

def test_DRLA(agent,env):
    test_env = DRLA_env(config)
    with torch.no_grad():
        state = test_env.reset()
        test_env.Model.load_state_dict(env.Model.state_dict())
        done = False
        while(not done):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            # print(reward)
            state = next_state
            if(done):
                welfare, workerList, acc = env.finalize('test')
                print('------------\nin test')
                print('finish. welfare : {}. workerList : {}\n------------'.format(welfare, workerList))
    return welfare, workerList
def train_DRLA():
    
    env = DRLA_env(config)

    N = config['N']
    agent = dqn_agent(N,config)
    agent.graph_opt = env.opt
    check_path(config['result_path'])
    check_path(config['model_path'])
    config_save = copy.deepcopy(config)
    config_save['device'] = 'cpu'
    json.dump(config_save, open(config['result_path'] + 'cfg.json','w'),indent=2)
    #######
    # 问题：sigma g选大了，welfare都是负数
    rewards = []
    welfares = []
    test_welfares = []
    best_welfare = -100
    result = {
        'ep_rwd': [],
        'acc': [],
        'test_rwd': []
        
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
                welfare, workerList, acc = env.finalize('train')
                
                if(welfare > best_welfare):
                    best_welfare = welfare
                    
                    ####
                    # save 2 model
                    agent.save(config['model_path']) 
                    torch.save(env.Model.state_dict(),config['model_path'] + 'GCN_model.pth')
                    
                if((epoch + 1) % config['test_every'] == 0):
                    test_welfare , _ = test_DRLA(agent,env)
                    test_welfares.append(test_welfare)
                    result['test_rwd'].append(float(test_welfare.detach().cpu().numpy()))
                    result['acc'].append(float(acc.detach().cpu().numpy()))
            cur_reward.append(reward)
            
        print(cur_reward)
        print('finish. welfare : {}, acc : {}, workerList : {}\n------------'.format(welfare,acc, workerList))
        rewards.append(ep_reward)
        welfares.append(welfare)
        result['ep_rwd'].append(float(welfare.detach().cpu().numpy()))
        
    # print('rewards : {}'.format(rewards))
        json.dump(result, open('{}result.json'.format(config['result_path']),'w'), indent=2)
    print('welfares : {}'.format(welfares))
    

if __name__ == "__main__":
    
    ## Data input
    train_DRLA()
    