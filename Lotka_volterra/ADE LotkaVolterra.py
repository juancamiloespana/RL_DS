# LIBRERÍAS
from LotkaVolterra import LotkaVolterraModel
import numpy as np
import random
import pandas as pd
import itertools
import time

#TIEMPO
inicio = time.time()

# FUNCIÓN - RANGOS DE FACTIBILIDAD -
def Cumple_Limites(parametros, limites):
    for i, valor in enumerate(parametros):
        if not (limites[i][0] <= valor <= limites[i][1]):
            return False
    return True

# PARÁMETROS DEL EXPERIMENTO
niveles_E = [0.1, 0.4, 0.6, 0.8]
niveles_fac = [[-0.001, 0, 0.001], [-0.01, 0, 0.01], [-0.1, 0, 0.1]]
repeticiones = 30

# TRATAMIENTOS
tratamientos = list(itertools.product(niveles_E, niveles_fac, range(1, repeticiones+1)))
random.shuffle(tratamientos) # Aleatorizar

resultados = []

# EJECUCIÓN
for E, fac, rep in tratamientos:
    P = [0.002, 0.04, 0.1, 0.0025]  #Valor inicial de los parámetros.
    Lim = [[0.0015, 0.0025], [0.03, 0.05], [0.05, 0.15], [0.002, 0.003]]  #Rango de factibilidad.
    A = len(fac)  #Número de acciones posibles.
    Q = np.zeros((A, A, A, A))
    Runs = 1000

    lotka_volterra_instance = LotkaVolterraModel(*P)
    R_inicial = lotka_volterra_instance.simulate()  #Solución inicial del modelo DS.

    y = [R_inicial]
    for i in range(Runs):
        P_previo = list(P)  # Guardo el valor de P previo
        lotka_volterra_instance = LotkaVolterraModel(*P_previo)
        R_previo = lotka_volterra_instance.simulate()

        if E < random.random():
            # Explotar
            max_val = np.amax(Q)
            result = np.where(Q == max_val)
            I, J, K, L = [result[i][0] for i in range(Q.ndim)]
        else:
            # Exploración
            idxs = [0] * Q.ndim
            I, J, K, L = [random.randint(0, A-1) for _ in idxs] #Genero una posición aleatoria.

        # Asigno un nuevo P de acuerdo a la posición elegida
        P[0] = P[0] * (1 + fac[I])
        P[1] = P[1] * (1 + fac[J])
        P[2] = P[2] * (1 + fac[K])
        P[3] = P[3] * (1 + fac[L])

        # Verifico que esta nueva solución P no viole los límites de factibilidad
        if Cumple_Limites(P, Lim):
            # Si sí es una solución factible calculo su recompensa
            lotka_volterra_instance = LotkaVolterraModel(*P)
            R_actual = lotka_volterra_instance.simulate()

            Q[I][J][K][L] += ((R_actual-R_previo)/R_previo)
            if R_previo >= R_actual:
                P = P_previo
        else:
            Q[I][J][K][L] += -100
            P = P_previo

        # Almacenar el resultado de la iteración
        y.append(R_previo)

        # Almacenar resultados
        resultados.append([E, fac[-1], rep, i+1, R_previo])

# RESULTADOS
df_resultados = pd.DataFrame(resultados, columns=['Nivel E', 'Nivel fac', 'Repetición', 'Corrida', 'y'])
df_resultados = df_resultados.sort_values(by=['Nivel E', 'Nivel fac', 'Repetición', 'Corrida'])
df_resultados.to_csv('Experimento_LotkaVolterra.csv', index=False)
print(df_resultados)

fin = time.time()
tiempo = (fin-inicio)/60
print(f"Tiempo total de ejecución: {tiempo:.3f} minutos")
