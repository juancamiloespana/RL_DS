import random
from typing import List
import pandas as pd
from random import sample
from random import random
import numpy as np

class Environment:
    def __init__(self):
        
        
        self.steps_left = 500
        self.k = 10
        self.df=pd.DataFrame()
        self.df['A'] = list(range(1,self.k+1))
        self.df['Q'] =0 ### Inicializar acumulado de valor de la acción
        self.df['N']= 0 ### Iniicalizar acumulado de número de veces de la acción
        self.df["mean_value"]=sample(list(range(50)),self.k) ##asignar un valor de media para cada brazo
        self.df["value_var"]=sample(list(range(1,15)),self.k) ##asignar un valor de varianza para cada brazo
    

    def get_observation(self) -> List[float]:
        return self.df

    def get_actions(self, agent) -> List[int]:       
        
        rnd=random()
        if(rnd >= agent.ep):
            max=self.df['Q'].max()
            bestAs=self.df['A'][self.df['Q']==max]
            a=bestAs.sample(1).values[0]
            a_index=self.df[self.df['A']==a].index.values[0] ## obtener indice de fila
            
        else:
            a=self.df['A'].sample().values[0]
            a_index=self.df[self.df['A']==a].index.values[0] ## obtener indice de fila

        return a, a_index

    def is_done(self) -> bool:
        return self.steps_left == 0

    def action(self, action: int) -> float:
        if self.is_done():
            raise Exception("Game is over")
        else:
            mu=self.df[self.df['A']==action[0]]['mean_value'].values[0]
            sig=self.df[self.df['A']==action[0]]['value_var'].values[0]
            R=np.random.normal(mu,sig,1)[0]
            
            N_A=self.df.loc[action[1],'N'] +1
            Q_A = self.df.loc[action[1],'Q'] ### Q acumulado hasta t-1
            Q_A=Q_A+((R-Q_A)/N_A)  ## Q actualizado t

            ## guardar valores de N y Q en tabla
            self.df.loc[action[1],'N'] = N_A
            self.df.loc[action[1],'Q'] = Q_A
            
            
        self.steps_left -= 1
        return R


class Agent:
    def __init__(self):
        self.total_reward = 0.0
        self.ep=0.10

    def step(self, env: Environment):
        current_obs = env.get_observation()
        actions = env.get_actions(self)
        reward = env.action(actions)
        self.total_reward += reward



if __name__ == "__main__":
    env = Environment()
    agent = Agent()

    while not env.is_done():
        agent.step(env)

    print("Total reward got: %.4f" % agent.total_reward)
    print("EL resumen de recompensa promedio por accción es: \n\n" , env.df)





