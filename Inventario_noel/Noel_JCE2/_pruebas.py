
import pandas as pd

from a_entorno2 import env_inventario
from b_agente2 import Agente
from c_RL_CE import RL_CE


import torch
from torch import nn

import numpy as np

import openpyxl

#### correr algoritmo ###


env=env_inventario()
agente= Agente(env)

h, dos=agente.get_action()

algo_rl=RL_CE(agente)


algo_rl.run_CE(BATCH_SIZE = 30, PERCENTILE = 70, iter_tot=10)




pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
algo_rl.resultados.head(10)
algo_rl.resultados.tail(60)
algo_rl.resultados.shape



algo_rl.resultados.to_excel('datos\\resultados.xlsx',index=False)



algo_rl.resultados['costo_tras'].sum()


lista=[1,3,4]

np_array=np.array(lista)



tenso=torch.FloatTensor([np_array])

obs=agente.env.observation


