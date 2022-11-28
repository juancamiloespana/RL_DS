import numpy as np
import gym
from gym import spaces
from gym.spaces import Discrete
from gym.spaces import MultiDiscrete
import datetime
from collections import namedtuple
import torch.optim as optim
import pandas as pd



### para red neuronal


import torch
from torch import nn

class inventario (gym.Env): 
    def __init__(
        self, 
        param={
        'max_tras':2000, ## máximo de pallet que se pueden trasladar
        'cap_cedi' :30000, ## definida en modelo
        'max_prod_r': 15000,
        'min_prod_r': 14000,
        'max_prod_n' : 20000,
        'min_prod_n': 15000,
        'dem_max_r':1500,
        'dem_min_r':1400,
        'dem_max_n':3000,
        'dem_min_n':150,
        'cost_be':1000,
        'penalidad_exceso':3500},
        steps=365         
    ):
        
        

        self.param=param
        self.penalizacion=1000000
        
        param['max_cedi']=param['cap_cedi']*2 ## un valor máximo que podría tomar la variable de inventario (se puede pasar de cap)
        param['max_be']=param['cap_cedi']*10 ## un valor máximo que podría tomar la variable de inventario (se puede pasar de cap)

        
        self.steps=steps
       
        
        self.action_space=spaces.Dict({
            'tras_be_r':Discrete(param['max_tras'], start=1),\
            'tras_be_n':Discrete(param['max_tras'], start=1),\
            'desp_n':Discrete(param['max_tras'], start=1),\
            'desp_r':Discrete(param['max_tras'], start=1), \
            'desp_be_r': Discrete(param['max_tras'], start=1), \
            'desp_be_n': Discrete(param['max_tras'], start=1)})
        
        
        self.observation_space=spaces.Dict({
            'cedi_level_r': Discrete(param['max_cedi'], start=0),\
            'cedi_level_n': Discrete(param['max_cedi'], start=0),\
            'be_level_r': Discrete(param['max_be'], start=0),\
            'be_level_n': Discrete(param['max_be'], start=0),\
            'prod_r': Discrete(param['max_be'], start=0),\
            'prod_n': Discrete(param['max_be'], start=0),\
            'dem_r': Discrete(param['max_be'], start=0),\
            'dem_n': Discrete(param['max_be'], start=0),\
            'exc_cedi': Discrete(param['max_cedi'], start=0),\
            'ocup_be': Discrete(param['max_be'], start=0)})
        
        self.observation=self.reset()
        
    def reset(self, seed=None, option=None):
        super().reset(seed=seed)
        
        self.resultados=pd.DataFrame()
        prod_r=self.np_random.integers(self.param['min_prod_r'], self.param['max_prod_r'],size=1,dtype=int)[0]
        prod_n = self.np_random.integers(self.param['min_prod_n'], self.param['max_prod_n'],size=1,dtype=int)[0]
        dem_r= self.np_random.integers(self.param['dem_min_r'], self.param['dem_max_r'],size=1,dtype=int)[0]
        dem_n= self.np_random.integers(self.param['dem_min_n'],self. param['dem_max_n'],size=1,dtype=int)[0]
        self.fecha=datetime.datetime(2022,1,1)
        
        observation={
            'cedi_level_r': 15000, \
            'cedi_level_n': 0,\
            'be_level_r': 0,\
            'be_level_n': 0,\
            'prod_r': prod_r,\
            'prod_n': prod_n, \
            'dem_r': dem_r,\
            'dem_n': dem_n,
            'exc_cedi':0,\
            'ocup_be':0           
            }
        
        self.observation=observation
        
        
        return observation
    
    def step(self, action):
        
        self.steps-=1
        
        
        terminated= self.steps<=0
        
        observation={}
        
        observation['cedi_level_r']= self.observation['cedi_level_r'] + self.observation['prod_r'] \
            - action['tras_be_r'] -action['desp_r']
    
        observation['cedi_level_n']= self.observation['cedi_level_n'] + self.observation['prod_n'] \
            - action['tras_be_n'] -action['desp_r']
        
        observation['be_level_r']= self.observation['be_level_r'] + action['tras_be_r'] \
            - action['desp_be_r']
            
        observation['be_level_n']= self.observation['be_level_n'] + action['tras_be_n'] \
            - action['desp_be_n']
            
        observation['prod_r'] = self.np_random.integers(self.param['min_prod_r'], self.param['max_prod_r'],size=1,dtype=int)[0]
        if self.fecha.month>= 11:
            observation['prod_n'] = self.np_random.integers(self.param['min_prod_n'], self.param['max_prod_n'],size=1,dtype=int)[0]
        else:
            observation['prod_n'] = 0
            
        observation['dem_r'] = self.np_random.integers(self.param['dem_min_r'], self.param['dem_max_r'],size=1,dtype=int)[0]
         
        if self.fecha.month >= 11:
            observation['dem_n'] = self.np_random.integers(self.param['min_prod_n'], self.param['max_prod_n'],size=1,dtype=int)[0]
        else:
            observation['dem_n'] = 0
            
        self.fecha+=datetime.timedelta(days=1)
        
###calculo reward ##

        ocup_cedi= observation['cedi_level_r'] +observation['cedi_level_n']  
        exceso=ocup_cedi -self.param['cap_cedi']
        observation['exc_cedi']  = np.max([0, exceso])
        observation['ocup_be'] = observation['be_level_r'] +observation['be_level_n']
        
        costo_be= np.max([self.param['cost_be'] * observation['ocup_be'] , observation['ocup_be']*self.penalizacion*-1])
        costo_exceso=observation['exc_cedi']*self.param['penalidad_exceso']
        
        tot_tras= sum(action.values())
        costo_tras=np.max([tot_tras - self.param['max_tras'],0])*self.penalizacion
        costo_cedi=np.max([0, ocup_cedi*-self.penalizacion] )
        
        #print('tot_tras:', tot_tras, 'costo_be:', costo_be, 'costo_exceso:',costo_exceso, 'costo_tras:', costo_tras,
         #    'costo_cedi:', costo_cedi)
        reward= costo_be+costo_exceso +costo_tras+costo_cedi
        
        
        self.observation=observation
        #### tabla de resultados ######

        
        self.resultados['fecha']=[self.fecha]
        self.resultados['cedi_level_r']= [self.observation['cedi_level_r']]
        self.resultados['cedi_level_n'] =self.observation['cedi_level_n']
        self.resultados['be_level_r'] =self.observation['be_level_r']
        self.resultados['be_level_n'] =self.observation['be_level_n']
        self.resultados['prod_r']=self.observation['prod_r']
        self.resultados['prod_n']=self.observation['prod_n']
        self.resultados['dem_r']=self.observation['dem_r']
        self.resultados['dem_n']=self.observation['dem_n']
        self.resultados['desp_r'] = action['desp_r']
        self.resultados['desp_n'] = action['desp_n']
        self.resultados['desp_be_r'] = action['desp_be_r']
        self.resultados['desp_be_n'] = action['desp_be_n']
        self.resultados['tras_be_r'] = action['tras_be_r']
        self.resultados['tras_be_n'] = action['tras_be_n']
        self.resultados['ocup_cedi'] = ocup_cedi
        self.resultados['ocup_be']=self.observation['ocup_be']
        self.resultados['exc_cedi']=self.observation['exc_cedi']
        self.resultados['tot_tras']=tot_tras
        self.resultados['costo_be']=costo_be
        self.resultados['costo_exceso']=costo_exceso
        self.resultados['costo_tras']= costo_tras
        self.resultados['tot_tras']=tot_tras
        self.resultados['tot_tras']=tot_tras
        self.resultados['tot_tras']=tot_tras
        self.resultados['tot_tras']=tot_tras
        self.resultados['costo_cedi']=costo_cedi
        self.resultados['reward']=reward
        ### faltan costos de transporte y de demanda insatisfecha ### 
        
         
        return observation, reward, terminated, self.resultados
               





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

 
Episode = namedtuple('Episode', field_names=['reward', 'steps','resultados'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])
 
def iterate_batches(env, net, batch_size): 
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs1 = env.reset()
    obs1=obs1.values()
    obs1=list(obs1)
    obs=np.array(obs1)
    resultados_ep=pd.DataFrame()
    e_count=1
  
    
    #sm = nn.Softmax(dim=1)
    while True:
        #print(n)
        #n+=1
        obs_v = torch.FloatTensor([obs])
        action_np = net(obs_v)
        action_np=action_np.data.numpy()[0]
        action=({ 'tras_be_r':action_np[0],\
            'tras_be_n':action_np[1],\
            'desp_n':action_np[2],\
            'desp_r':action_np[3], \
            'desp_be_r': action_np[4], \
            'desp_be_n':action_np[5]})
        
        
        ##act_probs = act_probs_v.data.numpy()[0]
        #action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, resultados_step = env.step(action)
        episode_reward += reward
        resultados_ep=pd.concat([resultados_ep,resultados_step])
        resultados_ep['e_count']=e_count
        
       
        step = EpisodeStep(observation=obs, action=action_np)
        episode_steps.append(step)
        if is_done:
            e_count+=1
            e = Episode(reward=episode_reward, steps=episode_steps, resultados=resultados_ep)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch  ## función generador, es como return, pero solo se puede usar una vez
                batch = []
        obs1 = next_obs
        obs1=obs1.values()
        obs1=list(obs1)
        obs=np.array(obs1)


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))
    resultados_bat=pd.DataFrame()

    train_obs = []
    train_act = []
    #reward, steps =batch[0][0], batch[0][1]
    for reward, steps, resultados in batch:
        #print(reward)
        if reward > reward_bound:
            resultados['filtrado']=1
            resultados_bat=pd.concat([resultados_bat,resultados])
            continue
        resultados['filtrado']=0
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))
        resultados_bat=pd.concat([resultados_bat,resultados])

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.FloatTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean, resultados_bat


if __name__ == "__main__":
    
    HIDDEN_SIZE = 128
    BATCH_SIZE = 16
    PERCENTILE = 70
    
    env = inventario()
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = len(env.observation_space)
    n_actions = len(env.action_space)

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.MSELoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    #writer = SummaryWriter(comment="-cartpole")
    resultados=pd.DataFrame()
    for iter_no, batch in enumerate(iterate_batches(
            env, net, BATCH_SIZE)):
        
        obs_v, acts_v, reward_b, reward_m, resultados_bat = \
            filter_batch(batch, PERCENTILE)
        resultados_bat['batche']=iter_no
        resultados=pd.concat([resultados,resultados_bat])
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m/1000000, reward_b))
        #writer.add_scalar("loss", loss_v.item(), iter_no)
        #writer.add_scalar("reward_bound", reward_b, iter_no)
        #writer.add_scalar("reward_mean", reward_m, iter_no)
        if iter_no > 20:
            print("Solved!")
            break
    #writer.close()
    
resultados=pd.DataFrame(columns=env.observation.keys())

env.

env= inventario()
len(obs_v)


import openpyxl

resultados.to_excel('resultados.xlsx')

resultados['fecha']=[env.fecha]
resultados.info()

resultados.iloc[-1]=pd.DataFrame(list(env.observation.values()))
env.reset()
env.observation.keys()
env.step(action) 


percentile=PERCENTILE

obs=uno.reset() 
obs1=obs.values()
obs2=list(obs1)
obs3=np.array(obs2)
torch.FloatTensor([obs3])


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70
n_action= 6

net=Net(10,HIDDEN_SIZE, 6)



uno.observation
action=({   'tras_be_r':10.5,\
            'tras_be_n':50,\
            'desp_n':1000,\
            'desp_r':3000, \
            'desp_be_r': 0, \
            'desp_be_n': 0})

uno.step(action) 


        
  observation={
            'cedi_level_r': 15000, \
            'cedi_level_n': 0,\
            'be_level_r': 0,\
            'be_level_n': 0,\
            'exc_cedi':0,\
            'ocup_be':0,\
            'fecha': [2022,1,1]            
            }
  
  
  fecha=datetime.datetime(observation['fecha'][0],observation['fecha'][1],observation['fecha'][2])
  
  fecha.year
  fecha.month
  fecha.day
  x=1
  y=2
  
  
  
  print('uno:',x, y)
  
  fecha+=datetime.timedelta(days=1)
  
  len(env.observation_space)
  
  del obs
  del obs_v