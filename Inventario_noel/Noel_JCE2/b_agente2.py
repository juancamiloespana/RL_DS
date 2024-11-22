
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
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)



class Agente:
    def __init__(self, env, HIDDEN_SIZE = 128):
               
        self.env=env
        
        obs_size = len(env.observation_space)
        n_actions = env.action_space.n
        #print(n_actions)
        
        self.net = Net(obs_size, HIDDEN_SIZE, n_actions)
        self.objective = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=0.01)
        self.resultados=pd.DataFrame()
        
    def get_action(self):
        
        env=self.env
        obs_dict = env.observation
        obs_dict_values=obs_dict.values()
        obs_list=list(obs_dict_values)
        obs_np=np.array(obs_list)
        obs_tensor = torch.FloatTensor(obs_np)
   
        sm=nn.Softmax(dim=0)
    
        probs = sm(self.net(obs_tensor))
        action_tensor=torch.argmax(probs, dim=0)
        action_np=action_tensor.data.numpy()*1

        return  obs_np, action_np
    
    def action(self):
        
        ### en este momento no tiene uso, por si se necesita alguna modificación a laacción de la red
        
        obs_np, action_np=self.get_action()
   
            
        return obs_np,  action_np
                    

        
    def learn(self, action_train, obs_train):
        
                      
        action_pred_tensor=self.net(obs_train)
        self.optimizer.zero_grad()
        
 
        action_train_long=action_train.long()
        loss_v = self.objective(action_pred_tensor,  action_train_long)
        loss_v.backward()
        self.optimizer.step()
        
        return (loss_v)
        
    
        
        
        
       
 