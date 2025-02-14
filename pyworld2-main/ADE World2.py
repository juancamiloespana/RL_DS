# LIBRERÍAS
import itertools
import json
import math
import random
import numpy as np
import os
import pandas as pd
import pyworld2
from pyworld2.utils import plot_world_variables, plt
import time
from tqdm import tqdm

#TIEMPO
inicio = time.time()

# FUNCIONES
def cargar_json(ruta):
    with open(ruta, "r") as file:
        return json.load(file)

def guardar_json(data, ruta):
    with open(ruta, "w") as file:
        json.dump(data, file, indent=4)

def update_json(data, brn1, nrun1, fc1, cign1, poln):
    for entry in data:
        if "BRN1" in entry:
            entry["BRN1"] = brn1 # BRN - Birth Rate Normal [fraction/year] Base run 0.028
        elif "NRUN1" in entry:
            entry["NRUN1"] = nrun1 # NRUN - Natural-Resource Usage Normal Base run 0.25
        elif "POLN" in entry:
            entry["POLN"] = poln # Pollution Normal [pollution units/person/year].
        elif "FC1" in entry:
            entry["FC1"] = fc1 # FC - Food Coefficient [] Base run 0.8
        elif "CIGN1" in entry:
            entry["CIGN1"] = cign1 # CIDN - Capital-Investment Discard Normal [fraction/year] Base run 0.03
    return data

def Cumple_Limites(parametros, limites):
    for i, valor in enumerate(parametros):
        if not (limites[i][0] <= valor <= limites[i][1]):
            return False
    return True

# PARÁMETROS DEL EXPERIMENTO
niveles_E = [0.6]
niveles_fac = [ [-0.01, 0, 0.01], [-0.1, 0, 0.1]]
repeticiones = 10

# Ruta del archivo JSON
input_file = os.path.join(os.path.dirname(__file__), "pyworld2", "functions_switch_default.json")

# TRATAMIENTOS
tratamientos = list(itertools.product(niveles_E, niveles_fac, range(1, repeticiones+1)))
random.shuffle(tratamientos) # Aleatorizar

resultados = []

# EJECUCIÓN
for E, fac, rep in tqdm(tratamientos):
    P = [0.028, 0.25, 0.8, 0.03, 0.5] #Valor inicial de los parámetros.
    Lim = [[0.02, 0.04], [0.1, 1.0], [0.6, 1.25], [0.02, 0.04], [0.1, 1.0]]  #Rango de factibilidad.
    A = len(fac)  #Número de acciones posibles.
    Q = np.zeros((A, A, A, A, A))
    Runs = 700
    
    # Cargar datos iniciales
    json_data = cargar_json(input_file)
    updated_data = update_json(json_data, *P)
    guardar_json(updated_data, "updated_data.json")

    # Inicializar el modelo
    w2_std = pyworld2.World2()
    w2_std.set_state_variables()
    w2_std.set_initial_state()
    w2_std.set_table_functions()
    w2_std.set_switch_functions("updated_data.json")
    w2_std.run()

    #Recompensa inicial
    R_inicial = w2_std.aveg_ql()  
    
    # Creación de vector para graficar el retorno
    y = [R_inicial]
    
    for i in range(Runs):
        P_previo = list(P)
        
        # Cargar datos y actualizar JSON
        json_data = cargar_json(input_file)
        updated_data = update_json(json_data, *P)
        guardar_json(updated_data, "updated_data.json")

        # Ejecutar el modelo
        w2_std = pyworld2.World2()
        w2_std.set_state_variables()
        w2_std.set_initial_state()
        w2_std.set_table_functions()
        w2_std.set_switch_functions("updated_data.json")
        w2_std.run()
        
        # Recompensa previa
        R_previo = w2_std.aveg_ql()  

       
        if E < random.random():
            # Explotar
            max_val = np.amax(Q)
            result = np.where(Q == max_val)
            I, J, K, L, M = [result[i][0] for i in range(Q.ndim)]
        else:
            # Explorar
            I, J, K, L, M = [random.randint(0, A - 1) for _ in range(Q.ndim)]

        # Asignar nuevo P de acuerdo a la posición elegida
        P[0] *= (1 + fac[I])
        P[1] *= (1 + fac[J])
        P[2] *= (1 + fac[K])
        P[3] *= (1 + fac[L])
        P[4] *= (1 + fac[M])

        # Verificar que la nueva solución P no viole los límites de factibilidad
        if Cumple_Limites(P, Lim):
            json_data = cargar_json(input_file)
            updated_data = update_json(json_data, *P)
            guardar_json(updated_data, "updated_data.json")

            w2_std = pyworld2.World2()
            w2_std.set_state_variables()
            w2_std.set_initial_state()
            w2_std.set_table_functions()
            w2_std.set_switch_functions("updated_data.json")
            w2_std.run()

            R_actual = w2_std.aveg_ql()  #Recompensa actual
            
            Q[I][J][K][L][M] += ((R_actual-R_previo)/R_previo)
            # Verifico que la nueva solución sea mejor que la mejor solución anterior, sino, me quedo con la solución mejor.
            if R_previo >= R_actual:
                P = P_previo 
        else:
            Q[I][J][K][L][M] += -100
            P = P_previo
            
        #Almacenar el resultado de la iteración
        y.append(R_previo)
        # Almacenar resultados
        resultados.append([E, fac[-1], rep, i+1, R_previo])

# RESULTADOS
df_resultados = pd.DataFrame(resultados, columns=['Nivel E', 'Nivel fac', 'Repetición', 'Corrida', 'y'])
df_resultados = df_resultados.sort_values(by=['Nivel E', 'Nivel fac', 'Repetición', 'Corrida'])
df_resultados.to_csv('Experimento_World2_2.csv', index=False)
print(df_resultados)

fin = time.time()
tiempo = (fin-inicio)/60
print(f"Tiempo total de ejecución: {tiempo:.3f} minutos")
#Tiempo total de ejecución: 3.327 minutos

#Tiempo total de ejecución: 0.631 minutos 2 CORRIDAS 2 REPETICIONES
#Tiempo total de ejecución: 0.682 minutos A100 gpu
# Tiempo total de ejecución: 0.485 minutos cpu
#Tiempo total de ejecución: 0.706 minutos L4
# Tiempo total de ejecución: 0.628 minutos T4 GPU
# Tiempo total de ejecución: 0.590 minutos v2-8 TPU
#Tiempo total de ejecución: 0.438 minutos
#Tiempo total de ejecución: 0.211 minutos
 
# Tiempo total de ejecución: 0.303 minutos reducido
#Tiempo total de ejecución: 0.209 minutos reducido v5e-1 TPU


