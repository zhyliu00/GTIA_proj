import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
import random
import copy
import dgl
import json
import matplotlib.pyplot as plt

res = {}
Ns = [30,50,70,90,110]
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

plt.rcParams['font.size'] = 24

for N in Ns:
    res[N] = json.load(open('./result/result_{}.json'.format(N),'r'))

    welfares = res[N]['test_rwd']
    ma_welfares = []
    for idx, v in enumerate(welfares):
        if(len(ma_welfares)==0):
            ma_welfares.append(v)
        else:
            ma_welfares.append(ma_welfares[-1]*0.9 + v * 0.1)
    res[N]['test_rwd']=ma_welfares
    

fig, ax = plt.subplots(1,1,figsize=(16,9))
# ax.set_xscale('log')
fig2, ax2 = plt.subplots(1,1,figsize=(16,9))
ax.set_xlabel('episodes')
ax2.set_xlabel('episodes')
ax.set_ylabel('Welfare')
ax2.set_ylabel('Accuracy')

for N in Ns:
    ax2.plot(np.array(list(range(len(res[N]['test_rwd'])) ))* 10,res[N]['acc'], label = 'N = {}'.format(N))
    ax.plot(np.array(list(range(len(res[N]['test_rwd'])) ))* 10  ,res[N]['test_rwd'], label = 'N = {}'.format(N))
ax.legend()
ax2.legend()
fig.savefig('Welfare.png')
fig2.savefig('Accuracy.png')
fig.savefig('Welfare.pdf')
fig2.savefig('Accuracy.pdf')