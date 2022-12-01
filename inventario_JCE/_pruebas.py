
from a_entorno import env_inventario
import numpy as np
import torch
from b_agente import Agente



env=env_inventario()
agente=Agente(env)


env.observation
action=agente.action()

env.step(action)
env.observation







