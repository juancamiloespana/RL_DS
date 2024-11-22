import numpy as np
import datetime
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Discrete
from gymnasium.spaces import Box
from gymnasium.spaces import MultiDiscrete


# #### Acciones seleccion
# ar 1 dentro de 16 políticas


# 	1: Priorizo Cedi N y R sin traslado
# 	2: Priorizo Cedi N y R  traslado 50/50
# 	3: Priorizo Cedi N y R traslado 100 n
# 	4: Priorizo Cedi N y R traslado 100 r
# 	5: Priorizo Cedi N sin traslado
# 	6: Priorizo Cedi N  traslado 50/50
# 	7: Priorizo Cedi N traslado 100 n
# 	8: Priorizo Cedi N traslado 100 r
# 	9: Priorizo Cedi R sin traslado
# 	10: Priorizo Cedi R  traslado 50/50
# 	11: Priorizo Cedi R traslado 100 n
# 	12: Priorizo Cedi R traslado 100 r
# 	13: Priorizo BE N y R sin traslado
# 	14: Priorizo BE N y R   traslado 50/50
#   15:Priorizo BE N y R traslado 100 n
#   16: Priorizo BE N y R traslado 100 r


Data = pd.read_excel('datos\\DemandaCEDI1.xlsx')
prod_reg = pd.DataFrame(Data, columns=['Producto_reg'])
prod_nav = pd.DataFrame(Data, columns=['Producto_nav'])
demand_reg = pd.DataFrame(Data, columns=['Demanda_reg'])
demand_nav = pd.DataFrame(Data, columns=['Demanda_nav'])


class env_inventario (gym.Env): 
    def __init__(
        self, 
        param={
        'max_tras':2000, ## máximo de pallet que se pueden trasladar
        'cap_cedi' :30000, ## definida en modelo
        'max_prod_r': 1500,
        'min_prod_r': 1400,
        'max_prod_n' : 2000,
        'min_prod_n': 1500,
        'dem_max_r':1500,
        'dem_min_r':1400,
        'dem_max_n':3000,
        'dem_min_n':150,
        'cost_be':2,
        'costo_tras':0.050,
        'costo_vp':8,
        'penalidad_exceso':4},
        steps=360         
    ):
        
        

        self.param=param
        
        param['max_cedi']=param['cap_cedi']*10 ## un valor máximo que podría tomar la variable de inventario (se puede pasar de cap)
        param['max_be']=param['cap_cedi']*10 ## un valor máximo que podría tomar la variable de inventario (se puede pasar de cap)


        self.steps_ini=steps
            
        
        self.action_space=spaces.Discrete(16, start=1)
        
        
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
        #prod_r=self.np_random.integers(self.param['min_prod_r'], self.param['max_prod_r'],size=1,dtype=int)[0]
        #prod_n=0
        prod_r= float(prod_reg.loc[0, 'Producto_reg'])
        prod_n = float(prod_nav.loc[0,'Producto_nav'])
       # dem_r= self.np_random.integers(self.param['dem_min_r'], self.param['dem_max_r'],size=1,dtype=int)[0]
        dem_r=float(demand_reg.loc[0,'Demanda_reg'])
        dem_n=float(demand_nav.loc[0,'Demanda_nav'])
        #dem_n= 0
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
        
        ###calculo reward antes de paso ##
        
        values_f_a=self.values_from_action(action, self.observation)
        
        
        
        costo_be=np.float64()
        costo_be=self.param['cost_be'] * self.observation['ocup_be'] 
        costo_exceso=self.observation['exc_cedi']*self.param['penalidad_exceso']
        
      
        
        vp=self.observation['dem_r']+self.observation['dem_n']-values_f_a['desp_r']-values_f_a['desp_be_r'] -values_f_a['desp_n'] -values_f_a['desp_be_n']
        costo_vp=vp*self.param['costo_vp']
        #print('tot_tras:', tot_tras, 'costo_be:', costo_be, 'costo_exceso:',costo_exceso, 'costo_tras:', costo_tras,
         #    'costo_cedi:', costo_cedi)
         
        tot_tras= values_f_a['tras_be_r']+values_f_a['tras_be_n']
        costo_tras=tot_tras*self.param['costo_tras']
        
        reward= costo_be+costo_exceso +costo_tras+costo_vp
        
        
       
  
        
        observation={}
        
        observation['cedi_level_r']= self.observation['cedi_level_r'] + self.observation['prod_r'] \
            - values_f_a['tras_be_r'] -values_f_a['desp_r']
    
        observation['cedi_level_n']= self.observation['cedi_level_n'] + self.observation['prod_n'] \
            - values_f_a['tras_be_n'] -values_f_a['desp_n']
        
        observation['be_level_r']= self.observation['be_level_r'] + values_f_a['tras_be_r'] \
            - values_f_a['desp_be_r']
            
        observation['be_level_n']= self.observation['be_level_n'] + values_f_a['tras_be_n'] \
            - values_f_a['desp_be_n']
            
        # observation['prod_r'] = self.np_random.integers(self.param['min_prod_r'], self.param['max_prod_r'],size=1,dtype=int)[0]
        # if self.fecha.month>= 10:
        #     observation['prod_n'] = self.np_random.integers(self.param['min_prod_n'], self.param['max_prod_n'],size=1,dtype=int)[0]
        # else:
        #     observation['prod_n'] = 0
            
        # observation['dem_r'] = self.np_random.integers(self.param['dem_min_r'], self.param['dem_max_r'],size=1,dtype=int)[0]
         
        # if self.fecha.month >= 11:
        #     observation['dem_n'] = self.np_random.integers(self.param['min_prod_n'], self.param['max_prod_n'],size=1,dtype=int)[0]
        # else:
        #     observation['dem_n'] = 0
        
        fila= self.steps_ini -self.steps
        
        
        observation['prod_r']= float(prod_reg.loc[fila,'Producto_reg'])
        observation['prod_n'] = float(prod_nav.loc[fila,'Producto_nav'])
      
        observation['dem_r']=float(demand_reg.loc[fila,'Demanda_reg'])
        observation['dem_n']=float(demand_nav.loc[fila,'Demanda_nav'])
            
        ocup_cedi= observation['cedi_level_r'] + observation['cedi_level_n']  
        exceso=ocup_cedi -self.param['cap_cedi']
        observation['exc_cedi']  = np.max([0, exceso])
        observation['ocup_be'] = observation['be_level_r'] +observation['be_level_n']
        

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
        self.resultados['desp_r'] = values_f_a['desp_r']
        self.resultados['desp_n'] = values_f_a['desp_n']
        self.resultados['desp_be_r'] = values_f_a['desp_be_r']
        self.resultados['desp_be_n'] = values_f_a['desp_be_n']
        self.resultados['tras_be_r'] = values_f_a['tras_be_r']
        self.resultados['tras_be_n'] = values_f_a['tras_be_n']
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
        self.resultados['costo_vp']=costo_vp
        self.resultados['reward']=reward
        self.resultados['action']=action
        ### faltan costos de transporte y de demanda insatisfecha ### 
        
        self.steps-=1
        self.terminated= self.steps<=0
        self.fecha+=datetime.timedelta(days=1)
        
        self.observation=observation
         
        return observation, reward, self.terminated, self.resultados
    
    def values_from_action(self, action, observation):
        
        values_f_a={}
        #print(action)
        ### condition for despachos n
        
        if action<=8:
     
            values_f_a['desp_n']=np.min([observation['dem_n'],observation['cedi_level_n']])
            rest_n=observation['dem_n'] - values_f_a['desp_n']
            values_f_a['desp_be_n'] =np.min([rest_n, observation['be_level_n']] )
        
        
        else:
            
            values_f_a['desp_be_n']=np.min([observation['dem_n'],observation['be_level_n']])
            rest_n=observation['dem_n'] - values_f_a['desp_be_n']
            values_f_a['desp_n'] =np.min([rest_n, observation['cedi_level_n']] )
            
 
            
      ### condition for despachos n
        
        
        if action in range(1,5) or action in range(9,13):
            
      
     
            values_f_a['desp_r']=np.min([observation['dem_r'],observation['cedi_level_r']])
            rest_n=observation['dem_r'] - values_f_a['desp_r']
            values_f_a['desp_be_r'] =np.min([rest_n, observation['be_level_r']] )
        
        
        else:
            
            values_f_a['desp_be_r']=np.min([observation['dem_r'],observation['be_level_r']])
            rest_n=observation['dem_r'] - values_f_a['desp_be_r']
            values_f_a['desp_r'] =np.min([rest_n, observation['cedi_level_r'] ])
            
        

        cedi_level_r= observation['cedi_level_r']- values_f_a['desp_r']
        cedi_level_n= observation['cedi_level_n']- values_f_a['desp_n']
        
        ### condition for traslados

        if action in [1,5,9,13]:
            values_f_a['tras_be_r'] =0
            values_f_a['tras_be_n'] =0
        
        elif action in [2,6,10,14]:
            
            values_f_a['tras_be_r'] =np.min([cedi_level_r, (self.param['max_tras']*0.5)])
            values_f_a['tras_be_n'] =np.min([cedi_level_n, (self.param['max_tras']*0.5)])
        
        elif action in [3,7,11,15]:
            
            values_f_a['tras_be_r'] =0
            values_f_a['tras_be_n'] =np.min([cedi_level_n, self.param['max_tras']])
      
        else: 
            values_f_a['tras_be_r'] =np.min([cedi_level_r, self.param['max_tras']])
            values_f_a['tras_be_n'] =0
                    
        
        
        return values_f_a
        
           
            
            
            
            
       
            
