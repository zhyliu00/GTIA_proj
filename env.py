import torch


class DRLA_env:
    def __init__(self,N,feats,cfg):
        self.feats = feats
        self.cfg = cfg
        self.g_hidden = cfg['g_hidden']
        self.stp = 0
    def reset(self):
        return self.feats

    def calc_reward(self):
        return torch.tensor([1])
    
    def calc_done(self):
        res = self.stp > 2
        return torch.tensor([res],dtype=torch.int16)
    
    def step(self, action):
        s = self.feats[:,2 * self.g_hidden]
        s[action] = 1
        self.stp+=1
        reward = self.calc_reward()
        # [1]
        done = self.calc_done()
        # [1]
        
        return self.feats, reward, done, {}
        