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
res = json.load(open('result.json','r'))

welfares = res['ep_rwd']
ma_welfares = []
for idx, v in enumerate(welfares):
    if(len(ma_welfares)==0):
        ma_welfares.append(v)
    else:
        ma_welfares.append(ma_welfares[-1]*0.9 + v * 0.1)


fig, ax = plt.subplots(1,1,figsize=(16,9))
ax.set_xscale('log')
ax.plot(ma_welfares)

fig.savefig('welfare.png')