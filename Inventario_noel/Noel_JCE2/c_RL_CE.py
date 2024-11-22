import numpy as np
from collections import namedtuple
import pandas as pd
import torch

 


class RL_CE:
    def __init__(self, agente):
        
        self.agente=agente
        self.env=agente.env
        
    
 
    def iterate_batches(self, batch_size): 
        
        
        Episode = namedtuple('Episode', field_names=['reward', 'steps','resultados'])
        EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])
        
        
        batch = []
        episode_reward = 0.0
        episode_steps = []
        self.env.reset() ## reiniciar entorno para empezar primer episode
        resultados_ep=pd.DataFrame()
        e_count=1
    
        while True:
            
          
            obs_np, action_np=self.agente.action()
            #print(action_np)
            
            next_obs, reward, is_done, resultados_step = self.env.step(action_np)
            episode_reward += reward
            resultados_ep=pd.concat([resultados_ep,resultados_step])
            resultados_ep['e_count']=e_count
        
            step = EpisodeStep(observation=obs_np, action=action_np)
            episode_steps.append(step)

            if is_done:
                e_count+=1
                e = Episode(reward=episode_reward, steps=episode_steps, resultados=resultados_ep)
                batch.append(e)
                episode_reward = 0.0
                episode_steps = []
                self.env.reset()
                resultados_ep=pd.DataFrame()
                if len(batch) == batch_size:
                    yield batch  ## función generador, es como return, pero solo se puede usar una vez
                    e_count=1
                    batch = []



    def filter_batch(self, batch, percentile):
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


    def run_CE(self, BATCH_SIZE = 20, PERCENTILE = 70, iter_tot=4 ):
           
        
        self.resultados=pd.DataFrame()
        
        for iter_no, batch in enumerate(self.iterate_batches(BATCH_SIZE)):
            print(iter_no)
            obs_train, action_train, reward_b, reward_m, resultados_bat = \
                self.filter_batch(batch, PERCENTILE)
                
            resultados_bat['batche']=iter_no
            self.resultados=pd.concat([self.resultados,resultados_bat])
            
            loss_v=self.agente.learn(action_train, obs_train)
        
            print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
                iter_no, loss_v.item(), reward_m, reward_b))
            
            if iter_no >= iter_tot:
                print("Solved!")
                break
            
            
        
        
        
        
        
        
        


        
