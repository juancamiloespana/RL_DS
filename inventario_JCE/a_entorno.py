import numpy as np
import datetime
import pandas as pd
import gym
from gym import spaces
from gym.spaces import Discrete
from gym.spaces import MultiDiscrete


class env_inventario (gym.Env): 
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

        self.steps_ini=steps
            
        
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
        prod_n = 0
        dem_r= self.np_random.integers(self.param['dem_min_r'], self.param['dem_max_r'],size=1,dtype=int)[0]
        dem_n= 0
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
        
        self.terminated=False
        self.steps=self.steps_ini
        return observation
    
    def step(self, action):
        
        self.steps-=1
        self.terminated= self.steps<=0
        
        observation={}
        
        observation['cedi_level_r']= self.observation['cedi_level_r'] + self.observation['prod_r'] \
            - action['tras_be_r'] -action['desp_r']
    
        observation['cedi_level_n']= self.observation['cedi_level_n'] + self.observation['prod_n'] \
            - action['tras_be_n'] -action['desp_n']
        
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
        
         
        return observation, reward, self.terminated, self.resultados
               
