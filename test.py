import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
import random
import copy
import dgl


a = torch.zeros((5,10))
s = a[:,3]
s[4] = 1

print(a)


a = torch.FloatTensor(1).uniform_(1e-5,1e-4)
b = 5**a
print(b)
# d = random.randint(2,10)
# d = torch.tensor([d])
# b = torch.tensor([a,a,d,d])
# print(b,b.shape)

# a = [1,2,3]
# b = [3,3,2]

# edges = a,b
# g = dgl.graph(edges)
# print(len(edges))
# print(g)
# J = 1
# K = 2
# N = 5

# A = torch.zeros((J,K,2,K))
# B = torch.zeros((N,2))

# C = torch.randn((5,10,1))

# print(C.shape)

# B = copy.deepcopy(C)
# C = torch.min(C,dim=1,keepdim=False)

# C[1].unsqueeze_(-1)
# print(C[1].shape)
# print(B.shape)
# print(B)
# print(C[1])
# # print(B.gather(1,C))
# print(B.gather(1,C[1]),B.gather(1,C[1]).shape)
# # D = torch.min(C[0],dim=0,keepdim=False)

# print(C[1].shape)

# A = torch.randn((3,10))
# a = torch.sum(A,dim=0)
# print(A,a)
# a = a.repeat(3,1)
# print(a.shape)
# print(a)

# a = torch.FloatTensor(1).uniform_(1e-5,1e-4)
# print(a)
# print(a.shape)

# C_total = list((range(1,101)))
# C_num = random.randint(2,6)
# C = random.sample(C_total,C_num)
# C_total = torch.tensor(C_total)
# C_num = torch.tensor(C_num)
# print(C,C_num)

# a = torch.tensor([1,2,3,5])
# b=  torch.tensor([5,3,3,3])
# d = torch.tensor([14])
# aa = np.intersect1d(a,b)
# cc = np.intersect1d(a,d)
# print(aa,cc)

# d_total = list(range(0,60000))
# d = random.randint(1,10)
# d_list = random.sample(d_total,d)
# d_total = torch.tensor(d_total)
# d = torch.tensor([d])
# d_list = torch.tensor(d_list)



# def Get_MNIST():
#     train_mnist =  torchvision.datasets.MNIST(root='./',train=True,download=True,transform=transforms.ToTensor())
#     test_mnist =   torchvision.datasets.MNIST(root='./',train=False,download=True,transform=transforms.ToTensor())
    
#     return train_mnist, test_mnist

# train,test = Get_MNIST()
# label_cnt = [0] * 10
# train_list = []
# for idx in d_list:
#     train_list.append(train[idx])
#     label_cnt[train[idx][1]]+=1

# train_X = None
# train_y = None
# for (X,y) in train_list:
#     X = X.unsqueeze(0)
#     y = torch.tensor([y])
#     if(train_X ==None):
#         train_X = X
#         train_y = y
#     else:
#         train_X = torch.cat([train_X,X],dim = 0)
#         train_y = torch.cat([train_y,y],dim = 0)
# print(train_X.shape, train_y.shape)


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
    
model = MNIST_model()
for param in model.parameters():
    print(param.shape)
# model1 = MNIST_model()
# model2 = MNIST_model()
# loss_fn = nn.CrossEntropyLoss()   
# # for name, params in model1.named_parameters():
# #     print(name , params.shape)
# opt = torch.optim.Adam(model1.parameters(), lr = 0.001)

# pred_y = model1(train_X)
# loss = loss_fn(pred_y, train_y)
# opt.zero_grad()
# loss.backward()
# opt.step()

# models = [MNIST_model(), MNIST_model()]
# model_final = MNIST_model()

# temp_params = {}
# flg = 1
# for model in models:
#     for name, param in model.named_parameters():
#         if(name not in temp_params.keys()):
#             temp_params[name] = param.clone()
#         else:
#             temp_params[name] += param

# for key,param in model_final.named_parameters():
#     temp_params[key] /= len(models)
#     param.data.copy_(temp_params[key])   
    
    
#     # print(param)

# for (name1,params1),(name2,params2),(name3,params3) in zip(models[0].named_parameters(),models[1].named_parameters(),model_final.named_parameters()):
#     print(name1,params1)
#     print(name2,params2)
#     print(name3,params3)
    

# print(model1.named_parameters()['out.2.bias'])
# print(model2.named_parameters()['out.2.bias'])
# print(model3.named_parameters()['out.2.bias'])

# model1.load_state_dict(sd3)



# for target_param, param in zip(model1.parameters(), model2.parameters()):
#     param.data.copy_(target_param.data)
    
# print(model1.state_dict())

# label_cnt = torch.tensor(label_cnt,dtype=torch.float32)
# label_cnt /= len(d_list)

# gnd_distb = [0.0987, 0.1124, 0.0993, 0.1022, 0.0974, 0.0903, 0.0986, 0.1044, 0.0975, 0.0992]
# sigma = 0.0
# for idx,v in enumerate(label_cnt):
#     sigma += torch.abs(label_cnt[idx]-gnd_distb[idx])
# sigma = torch.clip(sigma,max=1.2)
# print(sigma)

# for (x,y) in train_list:
#     print(x.shape,y)
# # print(train[1][0].shape)
# labels = [0] * 10
# for idx, (x,y) in enumerate(train):
#     # print(x,y)
#     # print(x.shape,y)
#     # labels.append(y)
#     labels[y] +=1
# print(labels)
# for idx,i in enumerate(labels):
#     labels[idx] /= 60000
# # print("{.3f}".format(labels))
# print ( ["{0:0.4f}".format(i) for i in labels])
# print(train[0][1].shape)
# print(train[1][1].shape)
# print(test)



# a = torch.zeros([10,5])
# b = torch.tensor([1])
# s = a[:,2]
# s[b]=1
# print(a)
# bas = []
# for i in range(10):
#     state = ([[2,2,2],[3,3,3]],torch.tensor([i]),torch.tensor(i+10),torch.randn((5,10)),0)
#     bas.append(state)
    
# batch = random.sample(bas, 4)
# # print(batch)
# state, action, reward, next_state, done = zip(*batch)
# print(action)
# if(isinstance(action[0],torch.Tensor)):
#     action = torch.stack(list(action),dim=0)
# print(action,action.shape)
# action = 
# print(torch.stack(action))
# state = list(state)
# print(state)
# state = torch.stack(state,dim=0)
# state = torch.stack(state,dim=0)
# print(state.shape)
# print(torch.tensor(state))
# C = torch.einsum('jkbd,nb->jknd',[A,B])
# print(C,C.shape)

# C = torch.exp(C)
# print(C,C.shape)

# D = torch.zeros((N,K))
# C[:,:,0,:] += 5
# C[:,1,:,:] +=6
# D[:,0] +=1
