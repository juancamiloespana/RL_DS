
import pandas as pd

from a_entorno import env_inventario
from b_agente import Agente
from c_RL_CE import RL_CE



#### correr algoritmo ###


env=env_inventario()
agente= Agente(env)

algo_rl=RL_CE(agente)
algo_rl.run_CE(BATCH_SIZE = 30, PERCENTILE = 70, iter_tot=10)




pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
algo_rl.resultados.head(10)
algo_rl.resultados.tail(60)
algo_rl.resultados.shape

import openpyxl

algo_rl.resultados.to_excel('resultados.xlsx',index=False)



algo_rl.resultados['costo_tras'].sum()


