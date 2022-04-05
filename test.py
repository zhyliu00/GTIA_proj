import torch
import random
import copy
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


a = torch.zeros([10,5])
b = torch.tensor([1])
s = a[:,2]
s[b]=1
print(a)
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
