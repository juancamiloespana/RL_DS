# Paquetes 
import gymnasium as gym
from collections import namedtuple
import numpy as np

from LotkaVolterra_JSJ import LotkaVolterraModel

import torch
import torch.nn as nn
import torch.optim as optim

import time

# Clase que construye la red
class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
            
        )

    def forward(self, x):
        x=self.net(x)
        return torch.tanh(x)/5  ## para que devuelva un porcentaje de cambio del parametro entre -0.2 y 0.2


# Clase agente
class Agent:

    def __init__(self, hs: int, bs: int, percent : int):
        
        self.hidden_size = hs
        self.batch_size = bs
        self.percentile = percent
        


    # Metodo para hacer un episodio
    def iterate_batches(self, env, net):
        #print("iterate batches inicio")
        
        # Clase que guarda la recompensa y el paso de un episodio
        Episode = namedtuple('Episode', field_names=['reward', 'steps'])
        # Clase que guarda la observacion y la accion del episodio
        EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action', 'reward'])

      
        batch = [] # Lista que guarda instancias de Episode ( reward y steps)
        self.episode_reward = 0.0 # Se inicializa la recompensa
        self.episode_steps = [] # Lista que guarda instancias de EpisodeStep (observación y acción)
        self.obs = env.reset() # Se genera una observación nueva

        #print("P inicial ", self.obs)
        i=1
        # Proceso de iteración para cada batch
        start_time2 = time.time()
        while True:
            start_time = time.time()
            
            
       
            
         
            self.obs=np.array(self.obs)
            self.obs_v = torch.FloatTensor(self.obs) # Se convierte la observación en un tensor
            self.act_probs_v = net(self.obs_v) # Se pasa la observación por la red y se calcula la acción
            self.act_probs = self.act_probs_v.data.numpy() # Conversión del tensor act_probs_v a un arreglo de probabilidades de acciones
            
            self.action = self.act_probs
      
          
            #print("Corrida =", i)
            
            #print('observacion actual')
            #print(self.obs)
                     
            #print(env.parameters)
           # print(env.parameters.dtype)
            # #print("acción")
            # print(self.action)
            # print(self.action.dtype)
            
            
    
            self.next_obs, reward, self.is_done = env.step(self.action, i) # Aplica la accion y calcula varias cosas
            # print("reward")
            # print(reward)
            # print("next_obs")
            # print(self.next_obs)
         
            
   
            #print( self.episode_reward)
            self.episode_reward += reward
        

            self.paso = EpisodeStep(observation=self.obs, action=self.action, reward=reward)
            self.episode_steps.append(self.paso)
            
            if self.is_done:  
                end_time2 = time.time()
                #print("tiuempo episodio")
                #print(end_time2 - start_time2)
                start_time2 = time.time()
                        
                self.e = Episode(reward=self.episode_reward, steps=self.episode_steps)
                #print(self.e)  
                batch.append(self.e)
                self.episode_reward = 0.0
                self.episode_steps = []
                self.next_obs = env.reset()
                #print(self.next_obs)
                
                #print('Episodio Numero: %d' % len(batch))
                #print("lenght batch", len(self.batch))
                #print("batch size ", self.batch_size)
                i=0
                if len(batch) == self.batch_size:
                    #print("termino batch")
                    yield batch
                    batch = []
                
            #print("termino")
            self.obs = self.next_obs
          
            i+=1
            #print(i)
            #print(len(batch))
            #print(self.batch_size)
            #end_time = time.time()
            #print("tiuempo corrida")
            #print(end_time - start_time)
            
    
    #  Metodo para filtrar los batches y quedar con los mejores
    def filter_batch(self, batch):

        self.rewards = list(map(lambda s: s.reward, batch))
        self.reward_bound = np.percentile(self.rewards, self.percentile)
        self.reward_mean = float(np.mean(self.rewards))


        self.train_obs = []
        self.train_act = []
        for reward, steps in batch:
            if reward < self.reward_bound:
                continue
            self.train_obs.extend(map(lambda step: step.observation, steps))
            self.train_act.extend(map(lambda step: step.action, steps))

        self.train_obs_v = torch.FloatTensor(self.train_obs)
        self.train_act_v = torch.FloatTensor(self.train_act)
        return self.train_obs_v, self.train_act_v, self.reward_bound, self.reward_bound, self.reward_mean
    
    def crossentropy(self, env, net, max_iter, lr):
        self.max_iter=max_iter
        #print(env.parameters)
        self.objective = nn.KLDivLoss()
        self.optimizer = optim.Adam(params=net.parameters(), lr=lr)
       # self.writer = SummaryWriter(comment="-SystemDynamics")
        self.batch_acum=[]
                   
        Batch_acum = namedtuple('Batch_acum', field_names=['n_batch', 'batch', ])
        
        for iter_no, batch in enumerate(self.iterate_batches(env, net)):
            end_time3 = time.time()
            start_time3 = time.time()
            #print("tiuempo batch")
            #print(end_time3 - start_time3) 
            batch_acum_i=Batch_acum(n_batch=iter_no, batch=batch)
            self.batch_acum.append(batch_acum_i)
            
            self.obs_v, self.acts_v, self.reward_b, self.reward_m, *_= \
                self.filter_batch(batch)
            self.optimizer.zero_grad()
            self.action_scores_v = net(self.obs_v)
           # print(self.action_scores_v.size())
           # print(self.acts_v.size())
            self.loss_v = self.objective(self.action_scores_v, self.acts_v)
            self.loss_v.backward()
            self.optimizer.step()
            #print(env.parameters)
            print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
                iter_no, self.loss_v.item(), self.reward_m, self.reward_b))
            #self.writer.add_scalar("loss", self.loss_v.item(), iter_no)
            #self.writer.add_scalar("reward_bound", self.reward_b, iter_no)
            #self.writer.add_scalar("reward_mean", self.reward_m, iter_no)
           
            if iter_no>=self.max_iter:
                print("Solved!")
                break
            
        #self.writer.close()

# Clase del entorno
class Environment:

    # Método inicializa la clase ambiente
    def __init__(self, runs: int, Lim):

  
   
        self.Runs = runs
        self.reward = 0
        self.Lim=Lim
        self.parameters = self.reset()
        
        
    # Método que resetea las observaciones
    def reset(self):
        
        self.parameters_reset = np.array([0.002, 0.035, 0.1, 0.025])
        #self.parameters_reset = np.array([np.random.uniform(low, high) for low, high in self.Lim])
        #self.parameters_reset=list(self.parameters_reset )
        self.parameters = self.parameters_reset
        return self.parameters
    
                
    def step(self, action: int, n_step: int):
        Lim=self.Lim
        self.termino = False
  
        P=np.array(self.parameters)
        #print(P)
        #print(action)
        P =  [1+action]*P
        P=P[0]
        for i in range(len(P)):
            if P[i] < Lim[i][0] or P[i] > Lim[i][1]:
                P[i]=self.parameters[i]
        self.parameters=P
        #self.parameters=list(self.parameters)
        #print(self.parameters)

        LV=LotkaVolterraModel(*self.parameters)
        self.reward =  LV.simulate()    
        #print(self.reward)
        self.termina= self.is_done(self.termino, n_step) 
    
        self.tuplex = (self.parameters, self.reward, self.termina)
        return self.tuplex


    def is_done(self, term : bool, ind: int):
        
        self.outlimits = term
        self.index = ind
        #print(self.Runs)
        #print(ind)
        #print(term)
        
        if self.outlimits == True or self.index == self.Runs:
            return True
        else:
            return False



### hiperparámetros de la red (agente)

n_actions=4 ## numero de salidas de la red, acciones (4 parametros lotkavolterra a modificar)
lr=0.01  ## learning rate de la red
hidden_size = 32 # Neuronas en la capa oculta

##hiperparametros del entorno
P = [0.002, 0.035, 0.1, 0.025] ## parametros iniciales
Lim = [[0.0010,0.004],[0.02,0.05],[0.05,0.2],[0.001,0.004]]

p=np.float64([0.002, 0.035, 0.1  , 0.025])
range(len(p))

##########
percentile = 75 # Percentil de descarte se borran este procentaje de episodios(los de menor reward promedio)

n_batches=10 ### número de batches
n_corridas =  50  ##numero de pasos por episodio ### tiempo por corrida 0.04 segundos
n_episodes = 10# Número de episodios por batch


tiempo_corrida=0.05

tiempo_total=tiempo_corrida*n_corridas*n_episodes*n_batches

tiempo_total/60
# tamaño de base para entrenar
np.round(n_episodes*(100-percentile+1)/100)*n_corridas



env = Environment(n_corridas, Lim)
obs_size=env.parameters.size
net = Net(obs_size,hidden_size, n_actions)
agent = Agent(hidden_size,n_episodes,percentile)
agent.crossentropy(env, net, n_batches, lr=lr)



params=[]
for param in net.parameters():
   params.append(param)
   break
   
params[0][-1].data.numpy()


total_params = sum(param.numel() for param in net.parameters())


obs=np.array([0.00236941, 0.0379374 , 0.08216001, 0.00250934])
obs_v = torch.FloatTensor([obs]) # Se convierte la observación en un tensor
act_probs_v = net(obs_v) # Se pasa la observación por la red y se calcula la acción
act_probs = act_probs_v.data.numpy()[0] # Conversión del tensor act_probs_v a un arreglo de probabilidades de acciones
act_probs




###### generar tabla de resultados

n_bat=[]
n_ep=[]
n_step=[]
reward_ep=[]
reward_step=[]
obs=[]
act=[]

batch_acum=agent.batch_acum


for i_batch, batch in agent.batch_acum:        
    for i_ep, ep in enumerate(batch):
        for i_step, paso in enumerate(ep.steps):
            n_bat.append(i_batch)
            n_ep.append(i_ep)
            n_step.append(i_step)
            reward_ep.append(ep.reward)
            reward_step.append(paso.reward)
            obs.append(paso.observation)
            act.append(paso.action)
            
           
         
import pandas as pd         
data={
'n_bat':n_bat,
'n_ep':n_ep,
'n_step':n_step,
'reward_ep':reward_ep,
'reward_step':  reward_step,
}   

data=pd.DataFrame(data)   

data1=pd.DataFrame(obs)
data1.columns=['P1','P2','P3','P4']
data2= pd.DataFrame(act)    
data2.columns=['PC_P1','PC_P2','PC_P3','PC_P4']   

resultados=pd.concat([data, data1, data2 ], axis=1)      
resultados['reward_ep']= resultados['reward_ep']/n_corridas
resultados.to_excel('resultados.xlsx')






# reward_ep=0

# n_batch=2
# for k in range(n_batch):
  
#     n_episodes=3
#     for j in range(n_episodes):
#         print()
#         print('Epsiodio numero: %d. reward_acum_ep: %d.' %(j,reward_ep))
#         print()
#         P=env.reset() ### parametros aleatorios generados cada que inicia episodio
#         reward_ep=0
        
#         n_corridas=4
#         for i in range(n_corridas):
           
#             print('    corrida numero: %d. reward_corrida: %d.' %(i,reward))
            
#             obs_t = torch.FloatTensor([P]) # Se convierte la observación en un tensor

#             actions_t = net(obs_t) # Se pasa la observación por la red y se calcula la acción
#             actions = act_probs_v.data.numpy()[0] #### se converte a numpy

#             P_nuevo=  [1+actions]*P
#             P=P_nuevo[0]
         
#             lv=LotkaVolterraModel(*P) ## total time son pasos de simulación
#             reward=lv.simulate()
#             reward_ep +=reward
            
#             if i==(n_corridas-1) and  j==(n_episodes-1):
#                 print()
#                 print("           terminó el batch, se eliminan episodios con reward bajitos y se rentrenar red neuronal con los otros")
#                 print()
#             ### cuando terminen las corridas, los episodios y los batch reentreno la red neuronal (net)
        
        
        





########

