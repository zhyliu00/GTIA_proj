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

parser.add_argument('--batch_size',type=int,default=256)
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
args = parser.parse_args()

config = {
    'device': torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu"),
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
    'epsilon_end' : 0.01,
    'epsilon_decay' : 1000,
    'memory_capacity' : 10000 , # capacity of Replay Memory
    'batch_size' : args.batch_size,
    'device' : torch.device(
        "cuda" if torch.cuda.is_available() else "cpu") , # check gpu
    'g_hidden':args.g_hidden,
    'J': args.J,
    'K': args.K,
    'N': args.N,
    'layers': args.layers
}
if(args.test_folder != None):
    curr_time = args.test_folder
config['result_path']= os.path.join(curr_path , "outputs/DRLA/" + curr_time + '/results/')  # path to save results
config['model_path']= os.path.join( curr_path ,"outputs/DRLA/" + curr_time + '/models/')  # path to save models

def A2graph(A):
    src_nodes = []
    tar_nodes = []
    for k,v in A.items():
        src = k
        for tar in v:
            src_nodes.append(src)
            tar_nodes.append(tar)
    G = dgl.graph((src_nodes,tar_nodes),device=config['device'])
    return G

def eval_GCN_ds(Models,data_dict,mode):
    # give model and iterable dataloader, return error
    test_loss = 0.0
    sm_test = 0
    test_preds = []
    test_gnds = []
    # test
    datas = {
        'graph_loader' : data_dict['loader'],
        'feat_total' : data_dict['X'],
        'label_total' : data_dict['y'][:,0,:,:],
        # feat : [B, T, N, F]
        'G' : data_dict['G'],
        # label : [B, T, N, 1]
        # label : [B, N, 1]
        # mask : [B, N]
        'mask_total' : data_dict['mask']
    }
    loss_fn = torch.nn.MSELoss(reduction='sum')

    preds_total = torch.zeros_like(datas['label_total'])
    volume_total = datas['feat_total'][:,-1,:,0].unsqueeze(2)
    # [B, N, 1]
    with torch.no_grad():
        for idx, (input_nodes,output_nodes,blocks) in enumerate(datas['graph_loader']):
           # feat : [B, T, N, F]
            pred, label, pred_t = calc_one_batch(Models, datas, (input_nodes,output_nodes,blocks), 'eval')
            preds_total[:,output_nodes,:]=pred_t
            loss = loss_fn(pred,label)
            sm_test += len(label)
            test_loss += loss.item()
            test_preds+=list(pred.reshape(-1))
            test_gnds+=list(label.reshape(-1))
    y_mean = data_dict['means']['y'][0]
    y_std = data_dict['stds']['y'][0]
    test_preds = torch.tensor(test_preds,dtype=torch.float32).to(config['device'])
    test_gnds = torch.tensor(test_gnds,dtype=torch.float32).to(config['device'])
    test_preds = test_preds * y_std + y_mean
    test_gnds = test_gnds * y_std + y_mean
    test_RMSE = calc_RMSE(test_preds,test_gnds)
    test_MAE = calc_MAE(test_preds,test_gnds)
    
    if(mode == 'test'):
        res = {
            'preds':preds_total,
            'label':datas['label_total'],
            'volume':volume_total,
            'means':data_dict['means'],
            'stds':data_dict['stds']
        }
        # for k,v in res.items():
        #     print(v.shape)
        print('saved')
        torch.save(res,os.path.join(config['result_path'],'{}_preds_labels.pt'.format(args.test_policy)))

    return test_loss / sm_test,test_RMSE,test_MAE
def test_vlocal(Models,Dataloaders):
    loss_arr = {
        # 'model':str(summary(Model,(20,config['hidden_states']))),
        'config':vars(args),
        "train_policy":args.train_policy,
        'test_policy':args.test_policy,
        'loss':{
            'test':None
        },
    }
    test_loss, test_RMSE, test_MAE = eval_GCN_ds(Models,Dataloaders['test'],'test')
    print('test loss: {}, RMSE : {}, MAE : {};'.format(test_loss,test_RMSE,test_MAE))
    loss_arr['loss']['test'] = 'test loss: {}, RMSE : {}, MAE : {};'.format(test_loss,test_RMSE,test_MAE)
    json.dump(loss_arr, open(config['result_path'] + 'test_result_{}.json'.format(args.test_policy), 'w'), indent=2)

def calc_one_batch(Models, datas, graphs, mode):
    for k,model in Models.items():
        model = model.to(config['device'])
        if(mode=='train'):
            model.train()
        else:
            model.eval()
    input_nodes, output_nodes, blocks = graphs
    # feat : [B, T, N, F]
    feat_total = datas['feat_total']
    label_total = datas['label_total']
    mask_total = datas['mask_total']
    # feat = feat_total[:,:,input_nodes,:]
    feat = feat_total[:,:,output_nodes,:]
    # feat : [B, T, N', F]

    # feat = feat.permute(0,2,1,3)
    # # feat : [B, N', T, F]
    # feat = feat.reshape(*feat.shape[:2],-1)
    # # feat : [B, N', T * F]

    # label : [B, N, 1]
    label = label_total[:,output_nodes,:]
    # label : [B, N', 1]

    # mask : [B, N]
    mask = mask_total[:,output_nodes]
    # mask : [B, N']
    
    
    pred = Models['Vlocal'](feat)
    # print('pred : {}, mask shape : {}, output_nodes : {}'.format(pred.shape,mask.shape,output_nodes))
    
    pred_mask = pred[mask]
    # pred : [B, N', 1]
    
    label = label[mask]

    return pred_mask, label, pred
        
def train_vlocal(Models,Dataloaders):
    model_str = ''
    for k, model in Models.items():
        model = model.to(config['device'])
        model_str += print_model(model) + '\n'
        
    best_val_loss = 999999.09
    # for k,v in Dataloaders.items():
    #     v.to(config['device'])
    # loss_fn = torch.nn.HuberLoss(reduction='sum',delta=3.0)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    opts = {}
    for k, model in Models.items():
        opts[k] = torch.optim.Adam(model.parameters(),lr = config['lr'])
    loss_arr = {
        'config':vars(args),
        'model':model_str,
        'loss':{
            'train':[],
            'val':[],
            'test':None
        },
        'RMSE':{
            'train':[],
            'val':[],
            'test':None
        },
        'MAE':{
            'train':[],
            'val':[],
            'test':None
        },
        'sizes':{
            'train': Dataloaders['train']['X'].shape,
            'val':Dataloaders['val']['X'].shape,
            'test':Dataloaders['test']['X'].shape
        },
        'avg_epoch_time':[]
    }
    epoch_times = []
    print('Start Training')
    cnt_notimporve = 0
    for epoch in range(config['max_epohs']):
        train_loss = 0.0
        sm_train = 0
        start_time = time.time()
        train_preds = []
        train_gnds = []
        # train
        datas = {
            'graph_loader' : Dataloaders['train']['loader'],
            'feat_total' : Dataloaders['train']['X'],
            'label_total' : Dataloaders['train']['y'][:,0,:,:],
            # feat : [B, T, N, F]
            'G' : Dataloaders['train']['G'],
            # label : [B, T, N, 1]
            # label : [B, N, 1]
            # mask : [B, N]
            'mask_total' : Dataloaders['train']['mask']
        }
                
        # Just for 1 step
        # print('feat: {}, label : {}, mask : {}'.format(feat_total.shape,label_total.shape,mask_total.shape))
        # if(spatial_layers==0):
        #     datas['graph_loader']= [(torch.arange(0,mask_total.shape[1]),torch.arange(0,mask_total.shape[1]),G)]

        for idx, (input_nodes,output_nodes,blocks) in enumerate(datas['graph_loader']):
            
            pred, label, _ = calc_one_batch(Models, datas, (input_nodes,output_nodes,blocks), 'train')
            
            loss = loss_fn(pred,label)
            
            sm_train += len(label)
            train_loss += loss.item()
            train_preds+=list(pred.reshape(-1))
            train_gnds+=list(label.reshape(-1))
            for k, opt in opts.items():
                opt.zero_grad()
            loss.backward()
            for k, opt in opts.items():
                opt.step()
        train_loss/=sm_train
        
        # 
        val_loss, val_RMSE, val_MAE = eval_GCN_ds(Models,Dataloaders['val'],'val')
        
        
        if(val_loss<best_val_loss):
            best_val_loss = val_loss
            for k, model in Models.items():
                torch.save(model.state_dict(),config['model_path'] + '{}.pth'.format(k))
            cnt_notimporve = 0
        else:
            cnt_notimporve += 1
        # if((epoch + 1) % config['save_every'] == 0):
        
        # with open('./cur_train_folder','w') as f:
        #     f.write(curr_time)
            
        end_time = time.time()
        
        y_mean = Dataloaders['train']['means']['y'][0]
        y_std = Dataloaders['train']['stds']['y'][0]
        train_preds = torch.tensor(train_preds,dtype=torch.float32).to(config['device'])
        train_gnds = torch.tensor(train_gnds,dtype=torch.float32).to(config['device'])

        train_preds = train_preds * y_std + y_mean
        train_gnds = train_gnds * y_std + y_mean

        train_RMSE = calc_RMSE(train_preds,train_gnds)
        train_MAE = calc_MAE(train_preds,train_gnds)
        
        loss_arr['loss']['train'].append(train_loss)
        loss_arr['loss']['val'].append(val_loss)
        loss_arr['RMSE']['train'].append(train_RMSE.item())
        loss_arr['RMSE']['val'].append(val_RMSE.item())
        loss_arr['MAE']['train'].append(train_MAE.item())
        loss_arr['MAE']['val'].append(val_MAE.item())
        
        cost_time = end_time-start_time
        epoch_times.append(cost_time)
        loss_arr['cost_time'] = epoch_times
        
        # print('in epoch {}/{}, cost time :{}, out train loss {}, val loss {}; in train loss {}, val loss {}'.format(epoch,config['max_epohs'],cost_time,train_loss_out,val_loss_out,train_loss_in,val_loss_in))
        
        print('in epoch {}/{}, cost time {}, train loss: {}, RMSE : {}, MAE : {}; val: loss: {}, RMSE : {}, MAE : {};'.format(epoch,config['max_epohs'],cost_time,train_loss,train_RMSE,train_MAE,val_loss,val_RMSE,val_MAE))
        
        if(epoch % config['save_every']==0):
            json.dump(loss_arr, open(config['result_path'] + 'loss.json', 'w'), indent=2)
            if(args.test_folder == None):
                with open('./cur_train_folder','w') as f:
                    f.write(curr_time)
        if(cnt_notimporve > 14):
            print("not imporve for {} epochs, end.".format(cnt_notimporve))
            json.dump(loss_arr, open(config['result_path'] + 'loss.json', 'w'), indent=2)
            break
    loss_arr['avg_epoch_time'] = np.mean(epoch_times)
    
    return loss_arr

def train_DRLA():
    
    
    N = 6
    config['N'] = N
    F = 5
    init_features = torch.randn((N,F))
    s = torch.zeros((N,1))
    v = torch.ones((N,args.g_hidden))
    
    edges = torch.tensor([0,0,2,3,4,5]),torch.tensor([1,3,5,4,1,1])
    e_weight = torch.tensor([0.1,0.2,0.3,0.4,0.5,0.6])
    g = dgl.graph(edges)
    g.edata['w'] = e_weight
    
    # here self loop weight is 0
    g = dgl.add_self_loop(g)
    # print(g.edata['w'])
    
    features = torch.ones((N,args.g_hidden))
    g.ndata['f'] = features
    Models = {
        # 5 his step, 4 each feature
        'GCN':GCN(input_size = args.g_hidden,hidden_size = args.g_hidden,layers = args.layers),
        'DQN':dqn_agent(args.N, config)
    }
    v_hat = Models['GCN'](g,g.ndata['f'],e_weight = g.edata['w'])
    
    print(v_hat,v_hat.shape)
    
    init_states = torch.cat([v_hat,v,s,init_features],dim = 1)
    print(init_states.shape)
    
    env = DRLA_env(N,init_states,config)
    agent = dqn_agent(N,config)
    
    state = env.reset()
    done = False
    while(not done):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        agent.memory.push(state, action, reward ,next_state, done)
        state = next_state
        agent.update()
        if(done):
            break
    
    

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
        