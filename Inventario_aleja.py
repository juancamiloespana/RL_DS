# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:25:46 2022

@author: alejandrs
"""
import numpy as np
import random as rn
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple

#Puedo hacer que la acción sea binaria para que active una fórmula o la otra
#Desición #1: Debo priorizar navidad o regular? 
#Desición #2: Debo priorizar cedi o bodega externa? 

#a=['Regular','Navidad','CEDI','Bodega']
#a=[1,0]

Hidden_size=100


"""-----------------------------------------------------------Red neuronal--------------------"""
class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential( #Permite crear las capas 
            nn.Linear(obs_size, hidden_size), #Unidad lineal
            nn.ReLU(), #Unidad no linear 
            nn.Linear(hidden_size, n_actions) #Esto arroja la probabilidad
            #nn.ReLU(),tira que el valor sea positivo, no una prob
        )

    def forward(self, x):
        return self.net(x)

"""------------------------------------------Environment------------------------------------------------"""


class cookies():
    def __init__(self,num=2, #n es el número de acciones que deberá tomar
                 da_min=1, da_max= 365, dt= 1, verbose=False): #Verbose imprime la nformación para debugging
        self.dt=dt
        self.da_min=da_min
        self.da_max= da_max
        self.verbose=True
        self.length = self.da_max - self.da_min
        self.n = int(self.length / self.dt)
        self.time = np.arange(self.da_min, self.da_max, self.dt)
        
        #Esta parte es del environment 
        
        self.num=num
        self.decision = np.full((self.num),0) #Vector de desiciones 
        self.obs = self.decision.tolist() # observaciones

        
    def init_cookies_variables(self,Cap_cedi= 30000, Cap_tras=2000): 
        self.Cap_cedi=np.full((self.n,),Cap_cedi)#1
        self.Cap_tras=np.full((self.n,),Cap_tras)#2
        self.Prod_nav=np.full((self.n,), 0)   #3
        self.Prod_reg=np.full((self.n,), 0)   #4
        self.Llegada_product=np.full((self.n,2), 0) #5
        
        self.Demand_prod_nav=np.full((self.n,), 0) #7
        self.Demand_prod_reg=np.full((self.n,), 0) #8
        self.Demand_por_vent=np.full((self.n,2), 0) #9
        self.Despach_bodeg=np.full((self.n,2), 0) #10
        self.Despach_desde_cedi=np.full((self.n,2), 0) #11
        self.CEDI=np.full((self.n,2), 0) #6
        
        self.Inv_proy=np.full((self.n,2), 0) #12
        self.Calc_exces_inv_cedi=np.full((self.n,), 0) #13
        self.A_trasladar=np.full((self.n,), 0)  #14
        self.Trans_Nav=np.full((self.n,), 0) #15
        self.Trans_reg=np.full((self.n,), 0) #16
        self.Exceso_trasla= np.full((self.n,2), 0) #17 
        self.Bodega_externa= np.full((self.n,2), 0) #18
        self.Ocup_bodeg_exter= np.full((self.n,), 0) #19
        self.Cost_diario=np.full((self.n,), 0, dtype=float) #20
        self.exces_invent=np.full((self.n,), 0) #21
        self.pena_exceso=np.full((self.n,), 0, dtype=float) #22
        self.Cost_tot_pena_exces_cedi= np.full((self.n,), 0, dtype=float) #23
        self.Cost_bodega_extern= np.full((self.n,), 0, dtype=float) #24
        self.Cost_total=np.full((self.n,), 0, dtype=float) #25
        
        
    #Ahora que se iniciaron todas las variables, se procedera con el bucle del tiempo
     
    def run_cookies(self,a,b): #Este es como el step, debe tomar una acción y actualizar el environment
        
        is_done=False
        
        
        self.n_moves+=1
        self.loopk(self.n_moves,a,b, alone=False)
        
        # Update observation
        self.decision[0]=1 #Regul o navid
        self.decision[1]=1 #Cedi o bedaga 
        self.obs = self.decision.tolist() 
              
    
            
        """for k_ in range(1, self.n): #Estas serían las iteraciones
            self.redo_loop = True
            while self.redo_loop:
                self.redo_loop = False
                if self.verbose:
                    print("go loop",k_)"""
                    
                
        if self.n_moves >= self.n:
          is_done = True
          
        return is_done
    
    
    def reset(self,a,b):
        
        self.init_cookies_variables()
        self.loop0(self,a,b,alone=False)
        
        self.decision = np.full((self.num),0) #Vector de desiciones 
        self.obs = self.decision.tolist() # observaciones
                    
#Funciones para actualizar las variables periodo a periodo
    
    
    def update_Prod_nav(self,i): #listo
        self.Prod_nav[i]=rn.randrange(1500,2000,1) #Cambie 0.1 por 1
        
    def update_Prod_reg(self,i): #Listo
        self.Prod_reg[i]=rn.randrange(1400,1500,1) #Cambie 0.1 por 1 
    
    def update_Llegada_product(self,i): #Listo
        self.Llegada_product[i]=np.array([self.Prod_reg[i],self.Prod_nav[i]])
        
    def update_Demand_prod_nav(self,i):
        self.Demand_prod_nav[i]=rn.randrange(150,3000,1) #Cambie 0.1 por 1
        
    def update_Demand_prod_reg(self,i):
        self.Demand_prod_reg[i]=rn.randrange(1400,1500,2) #Cambie 0.1 por 2
              
    def update_Demand_por_vent(self,i): #Listo
        self.Demand_por_vent[i]=np.array([self.Demand_prod_reg[i],self.Demand_prod_nav[i]])
        
    def update_Despach_bodeg(self,i,b): #Listo
        if self.decision[1]==0:
            self.Despach_bodeg[i][0]=min(self.Bodega_externa[i][0],(self.Demand_por_vent[i][0]-self.Despach_desde_cedi[i][0]))
            self.Despach_bodeg[i][1]=min(self.Bodega_externa[i][1],(self.Demand_por_vent[i][1]-self.Despach_desde_cedi[i][1]))
        else: 
            self.Despach_bodeg[i][0]=min(self.Bodega_externa[i][0],(self.Demand_por_vent[i][0]))
            self.Despach_bodeg[i][1]=min(self.Bodega_externa[i][1],(self.Demand_por_vent[i][1]))
    
    def update_Despach_desde_cedi(self,i,b): #Listo #0 prefiere CEDI y 1 prefiere bodega externa
        if self.decision[1]==0:
            self.Despach_desde_cedi[i][0]=min( self.CEDI[i][0],(self.Demand_por_vent[i][0]))
            self.Despach_desde_cedi[i][1]=min( self.CEDI[i][1],(self.Demand_por_vent[i][1]))
        else: 
            self.Despach_desde_cedi[i][0]=min( self.CEDI[i][0],(self.Demand_por_vent[i][0] - self.Despach_bodeg[i][0]))
            self.Despach_desde_cedi[i][1]=min( self.CEDI[i][1],(self.Demand_por_vent[i][1] - self.Despach_bodeg[i][1]))
        
    def update_CEDI(self,i):
        self.CEDI[i] = ( (self.Llegada_product[i] - self.Despach_desde_cedi[i] )*self.dt ) +self.CEDI[i-1]
       
    def update_Inv_proy(self,i): #Listo
        self.Inv_proy[i][0]=(self.Llegada_product[i][0]-self.Despach_desde_cedi[i][0])+self.CEDI[i][0]
        self.Inv_proy[i][1]=(self.Llegada_product[i][1]-self.Despach_desde_cedi[i][1])+self.CEDI[i][1]
   
    def update_Calc_exces_inv_cedi(self,i): #listo
        self.Calc_exces_inv_cedi[i]= max((self.Inv_proy[i].sum() - self.Cap_cedi[i]),0)
        
    def update_A_trasladar(self,i): #listo
        self.A_trasladar[i]=min(self.Calc_exces_inv_cedi[i],self.Cap_tras[i])
        
    def update_Trans_Nav(self,i,a): #Listo
        if self.decision[0]==0: #Se prefiere Producto navideño
            self.Trans_Nav[i]=  min(self.A_trasladar[i],self.CEDI[i][1])
        else:
            self.Trans_Nav[i]=self.A_trasladar[i]-self.Trans_Reg[i]

      
    def update_Trans_reg(self,i,a): #Listo
        if self.decision[0]==0:
            self.Trans_reg[i]=self.A_trasladar[i]-self.Trans_Nav[i] 
        else:
            self.Trans_reg[i]=min(self.A_trasladar[i],self.CEDI[i][0])
        
    def update_Exceso_trasla(self,i): #Listo
        self.Exceso_trasla[i]=np.array([self.Trans_reg[i],self.Trans_Nav[i]])   
        
    def update_Bodega_externa(self,i):
        self.Bodega_externa[i] = ((self.Exceso_trasla[i] - self.Despach_bodeg[i] ) * self.dt) +self.Bodega_externa[i-1]
        
    def update_Ocup_bodeg_exter(self,i):
        self.Ocup_bodeg_exter[i]=self.Bodega_externa[i].sum()  
         
    def update_Cost_diario(self,i):
         self.Cost_diario[i]=(self.Ocup_bodeg_exter[i]*1000)
    
    def update_exces_invent(self,i):
        self.exces_invent[i]=max(self.CEDI[i].sum() - self.Cap_cedi[i], 0)
         
    def update_pena_exceso(self,i):
        self.pena_exceso[i]=(self.exces_invent[i]*3500)
     
    def update_Cost_tot_pena_exces_cedi(self,i):
        self.Cost_tot_pena_exces_cedi[i]= ( ((self.pena_exceso[i] * self.dt ) + self.Cost_tot_pena_exces_cedi[i-1]))
    
    def update_Cost_bodega_extern(self,i):
        self.Cost_bodega_extern[i] = ( (self.Cost_diario[i]*self.dt)  + self.Cost_bodega_extern[i-1])
    
    def update_Cost_total(self,i):
        self.Cost_total[i]= (self.Cost_tot_pena_exces_cedi[i]+self.Cost_bodega_extern[i])
     
        
    def loop0(self,a,b,alone=False):
        
        self.update_Prod_nav(0)   #3
        self.update_Prod_reg(0)   #4
        self.update_Llegada_product(0)  #5
        
        self.update_Demand_prod_nav(0)  #7
        self.update_Demand_prod_reg(0)  #8
        self.update_Demand_por_vent(0)  #9
        
        self.CEDI[0]=np.array([15000,0]) #6
        self.update_Despach_desde_cedi(0,b) #11
          
        
        self.update_Inv_proy(0)  #12
        self.update_Calc_exces_inv_cedi(0) #13
        self.update_A_trasladar(0) #14
        
        if self.decision[0]==0:
            self.update_Trans_Nav(0,a)  #15
            self.update_Trans_reg(0,a)  #16
        else:
            self.update_Trans_reg(0,a)
            self.update_Trans_Nav(0,a)
            
        self.update_Exceso_trasla(0)  #17 
        self.update_Bodega_externa(0)  #18
        self.update_Despach_bodeg(0,b)  #10
        
        self.update_Ocup_bodeg_exter(0)  #19
        self.update_Cost_diario(0)  #20
        self.update_exces_invent(0) #21
        self.update_pena_exceso(0)  #22
        self.update_Cost_tot_pena_exces_cedi(0) #23
        self.update_Cost_bodega_extern(0)  #24
        self.update_Cost_total(0) #25
        

        
    def loopk(self,i,a,b, alone=False):
        
        self.update_Prod_nav(i)  #3
        self.update_Prod_reg(i)   #4
        self.update_Llegada_product(i) #5
        
        self.update_Demand_prod_nav(i) #7
        self.update_Demand_prod_reg(i) #8
        self.update_Demand_por_vent(i)  #9
        
        self.update_CEDI(i) #6
        
        self.update_Despach_desde_cedi(i,b) #11        
        self.update_Inv_proy(i)  #12
        self.update_Calc_exces_inv_cedi(i)  #13
        self.update_A_trasladar(i)  #14
        
        if self.decision[0]==0:
            self.update_Trans_Nav(i) #15
            self.update_Trans_reg(i)  #16
        else:
            self.update_Trans_Nav(i,a) #15
            self.update_Trans_reg(i,a)#16
            
        self.update_Exceso_trasla(i)  #17 
        
        self.update_Bodega_externa(i) #18
        self.update_Despach_bodeg(i,b)  #10 #No ´sé si se deba correr antes que CEDI

        self.update_Ocup_bodeg_exter(i) #19
        self.update_Cost_diario(i) #20
        self.update_exces_invent(i) #21
        self.update_pena_exceso(i)#22
        self.update_Cost_tot_pena_exces_cedi(i)  #23
        self.update_Cost_bodega_extern(i)  #24
        self.update_Cost_total(i)  #25
        
"""--------------------------------------Agent----------------------------------------------"""
#El agente es el encargado de tomar la desición

class AGENT():
    def __init__(self, softmax, e):    
        self.a = -99 # Define null action
        self.b=-99# Define null action
        self.softmax = softmax
        self.e = 0.001


    def choose_action(self, cook, net):
        '''Obtains and observation from the environment and chooses and 
        action based on the probabilities given for a neural network'''  
        cook=cookies()
        p = rn.random() # Generate random number
        
          
        if p < self.e:
          # Randomly select an action
          self.a = rn.randint(0,1)
          self.b= rn.randint(0,1)
          
        else:
          # Take greedy action chosing the one with highest probability
          obs_v = torch.FloatTensor([cook.obs]) #Cambiar las observaciones
          print(obs_v)
          act_probs_v = self.softmax(net(obs_v))
          act_probs = act_probs_v.data.numpy()[0]      
          self.a = np.random.choice(len(act_probs), p=act_probs)      
          self.b = np.random.choice(len(act_probs), p=act_probs) 
          
        return self.a,self.b        
    
"""--------------------------------------Iteramos los batches--------------------------------"""     
# namedtuples se utiliza para guardar los pasos de cada episodio y los episodios 
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'a', 'b'])
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
def iterate_batches(cook, agent, net, batch_size):
  '''Generate batches of knapsack solutions. Each solution consist of the 
     steps = [observation, actions] taken to solve the problem. 
     The batch collects several solutions (batch_size) '''  
  batch = []
  episode_reward = 0.0
  episode_steps = []
  sm = nn.Softmax(dim=1)
  cook=cookies()
  a = agent.choose_action(cook, net)    
  b = agent.choose_action(cook, net)
  cook.reset(a=a,b=b)

  # Repeat until enough solutions are built
  while True:
    a = agent.choose_action(cook, net)    
    b = agent.choose_action(cook, net) 
    
    is_done = cook.run(a,b)
    step = EpisodeStep(observation=cook.obs, a=a, b=b)
    episode_steps.append(step) 
    # if a solutions is complete, it reset the environment to start a new solution   
    if is_done:
      e = Episode(reward= -cook.Cost_total, steps=episode_steps)
      batch.append(e)
      cook.reset(a=a,b=b)
      episode_steps = []
      # If enough solutions are generated it returns the batch
      if len(batch) == batch_size:
        yield batch #yield es como un return pero para generadores(como una lista porque puede ser recorrida con iteradores) pero no se está guardando el resultado
        batch = []


"""-------------------------------------Filtramos los batches-------------------------------"""
def filter_batch(batch, percentile):
    '''Selects the best solutions given a percentile  ''' 
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    position_best = np.argmax(rewards)
    reward_max = rewards[position_best]    

    train_obs = []
    train_act = []
    train_actb = []
    
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda run_cookies: run_cookies.observation, steps))
        train_act.extend(map(lambda run_cookies: run_cookies.a, steps))
        train_actb.extend(map(lambda run_cookies: run_cookies.b, steps))
        
    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    train_actb_v = torch.LongTensor(train_actb)
    
    return train_obs_v, train_act_v,train_actb_v, reward_bound, reward_max

"""--------------------------------------Prueba---------------------------------------------"""

# Parameters
HIDDEN_SIZE = 100
BATCH_SIZE = 16
PERCENTILE = 70
N_EPISODES = 365
EPSILON = 0.001

# Lists to save the observed solutions
incunbent_hist =[0]
incunbent = -9999999999999
best_list = [0]

# Create neural network 
net = Net(2, HIDDEN_SIZE, 2)
objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=net.parameters(), lr=0.01)
sm = nn.Softmax(dim=1)

# Create  agent and environment
env = cookies()
ag = AGENT(sm, EPSILON)
       
#Iterate over batches. Each time a batch is created the neural network is
# updated
for iter_no, batch in enumerate(iterate_batches(env, ag, net, BATCH_SIZE)):
  
  # Filters the batch
  obs_v, acts_v,actb_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
  
  # Updates the incumbent (best solution)
  if reward_m > incunbent:
    incunbent = reward_m
  incunbent_hist.append(incunbent)
  best_list.append(reward_m)

  # Update network
  optimizer.zero_grad()
  action_scores_v = net(obs_v)
  loss_v = objective(action_scores_v, acts_v, actb_v)
  loss_v.backward()
  optimizer.step()
  
  # stops if the number of episodes is reached
  if iter_no >= N_EPISODES:
    break

import plotly.graph_objects as go


fig = go.Figure()
x = list(range(N_EPISODES))
fig.add_trace(go.Scatter(x = x, y = incunbent_hist, name= "incumbent"))
fig.add_trace(go.Scatter(x = x, y = best_list, name= "best_episode"))
 
fig.show()
  
def hello_cookies():
    
    Cookies = cookies()
    Cookies.init_cookies_variables()
    Cookies.run_cookies()
    
    print('Cost_total',Cookies.Cost_total)
     

if __name__ == "__main__":
    hello_cookies()
    