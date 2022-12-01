
import numpy as np
import datetime
from collections import namedtuple
import torch.optim as optim
import pandas as pd
import torch
from torch import nn


class Net(nn.Module): ###herencia de propiedads de nn.Module
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__() ### hereda métodos y propiedades de la clase padre
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Agente:
    def __init__(self, env, HIDDEN_SIZE = 128):
               
        self.env=env
        
        obs_size = len(env.observation_space)
        n_actions = len(env.action_space)
        
        self.net = Net(obs_size, HIDDEN_SIZE, n_actions)
        self.objective = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=0.01)
        self.resultados=pd.DataFrame()
        
    def get_action(self):
        
        env=self.env
        obs_dict = env.observation
        obs_dict_values=obs_dict.values()
        obs_list=list(obs_dict_values)
        obs_np=np.array(obs_list)
        obs_tensor = torch.FloatTensor([obs_np])
        
        action_tensor = self.net(obs_tensor)
        action_np=action_tensor.data.numpy()[0]
        action_dict=({ 'tras_be_r':action_np[0],\
            'tras_be_n':action_np[1],\
            'desp_n':action_np[2],\
            'desp_r':action_np[3], \
            'desp_be_r': action_np[4], \
            'desp_be_n':action_np[5]})
        
        return  obs_np, action_dict
    
    def action(self):
        
        obs_np, action_dict=self.get_action()
        obs_dict=self.env.observation
        
        if action_dict['desp_r']>=obs_dict['cedi_level_r']:
            action_dict['desp_r'] = obs_dict['cedi_level_r']
        
        if  action_dict['tras_be_r'] >= obs_dict['cedi_level_r'] - action_dict['desp_r']:
            action_dict['tras_be_r']= obs_dict['cedi_level_r'] -action_dict['desp_r']
            
        if action_dict['desp_n']>=obs_dict['cedi_level_n']:
            action_dict['desp_n'] = obs_dict['cedi_level_n']
        
        if  action_dict['tras_be_n'] >= obs_dict['cedi_level_n'] -action_dict['desp_n']:
            action_dict['tras_be_n']= obs_dict['cedi_level_n'] -action_dict['desp_n']
        
        if action_dict['desp_be_r']>=obs_dict['be_level_r']:
            action_dict['desp_be_r'] = obs_dict['be_level_r']
                
        if action_dict['desp_be_n']>=obs_dict['be_level_n']:
            action_dict['desp_be_n'] = obs_dict['be_level_n']
        
        if action_dict['desp_r'] + action_dict['desp_be_r']> obs_dict['dem_r']:
        
            total_desp= action_dict['desp_r'] + action_dict['desp_be_r']
            prop_cedi= action_dict['desp_r']/total_desp
            prop_be= 1-prop_cedi
            action_dict['desp_r'] = obs_dict['dem_r']*prop_cedi
            action_dict['desp_be_r'] = obs_dict['dem_r']*prop_be
            
        if action_dict['desp_n'] + action_dict['desp_be_n']> obs_dict['dem_n']:
            print(action_dict['desp_n'], action_dict['desp_be_n'])
            total_desp= action_dict['desp_n'] + action_dict['desp_be_n']
            prop_cedi= action_dict['desp_n']/total_desp
            prop_be= 1-prop_cedi
            action_dict['desp_n'] = obs_dict['dem_n']*prop_cedi
            action_dict['desp_be_n'] = obs_dict['dem_n']*prop_be
            
        
        action_dict_values=action_dict.values()
        action_list=list(action_dict_values)
        action_np=np.array(action_list)
        
            
        return obs_np, action_dict, action_np
                    


        
    def learn(self, action_train, obs_train):
        
        env=self.env
                
        action_pred_tensor=self.net(obs_train)
        self.optimizer.zero_grad()
        loss_v = self.objective(action_pred_tensor, action_train)
        loss_v.backward()
        self.optimizer.step()
        
    
        
        
        
       
 