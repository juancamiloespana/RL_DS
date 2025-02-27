# LIBRERÍAS
import json
import math
import random
import numpy as np
import os
import pyworld2
from pyworld2.utils import plot_world_variables, plt
import time
from tqdm import tqdm


# FUNCIONES
def cargar_json(ruta):
    with open(ruta, "r") as file:
        return json.load(file)

def guardar_json(data, ruta):
    with open(ruta, "w") as file:
        json.dump(data, file, indent=4)

def update_json(data, brn1, nrun1, fc1, cidn1, poln):
    for entry in data:
        if "BRN1" in entry:
            entry["BRN1"] = brn1 # BRN - Birth Rate Normal [fraction/year] Base run 0.028
        elif "NRUN1" in entry:
            entry["NRUN1"] = nrun1 # NRUN - Natural-Resource Usage Normal Base run 0.25
        elif "POLN" in entry:
            entry["POLN"] = poln # Pollution Normal [pollution units/person/year].
        elif "FC1" in entry:
            entry["FC1"] = fc1 # FC - Food Coefficient [] Base run 0.8
        elif "CIDN1" in entry:
            entry["CIDN1"] = cidn1 # CIDN - Capital-Investment Discard Normal [fraction/year] Base run 0.03
    return data

def Cumple_Limites(parametros, limites):
    for i, valor in enumerate(parametros):
        if not (limites[i][0] <= valor <= limites[i][1]):
            return False
    return True

# Configuración inicial
E = 0.4  # Parámetro de exploración
fac = [-0.01, 0, 0.01]  # Posibles cambios en los parámetros
P = [0.028, 0.25, 0.8, 0.03, 0.5]  # Valores iniciales de los parámetros
Lim = [[0.02, 0.04], [0.1, 1.0], [0.6, 1.25], [0.02, 0.04], [0.1, 1.0]]  # Rangos de factibilidad
A = len(fac)  # Número de acciones posibles
Q = np.zeros((A, A, A, A, A))  # Matriz Q del método K-Armed Bandit
Runs = 500  # Número de corridas

# Ruta del archivo JSON
input_file = os.path.join(os.path.dirname(__file__), "pyworld2", "functions_switch_default.json")

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

R_inicial = w2_std.aveg_ql()  # Recompensa inicial
print("Parámetros iniciales =", P)
print("Recompensa inicial =", round(R_inicial, 5))

# Creación de vector para graficar el retorno
y = [R_inicial]

# Proceso de explorar-explotar
for i in tqdm(range(Runs)):
    P_previo = list(P)  # Guardar el valor de P previo

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

    R_previo = w2_std.aveg_ql()  # Recompensa previa

    # Se grafica las ganancias
    y.append(R_previo)

    # Exploración o explotación
    if E < random.random():
        # Opción explotar
        max_val = np.amax(Q)  # Identificar el máximo valor
        result = np.where(Q == max_val)  # Encontrar la posición del máximo valor
        I, J, K, L, M = [result[i][0] for i in range(Q.ndim)]  # Guardar la posición del máximo valor
    else:
        # Proceso de exploración
        I, J, K, L, M = [random.randint(0, A - 1) for _ in range(Q.ndim)]  # Generar posición aleatoria

    # Asignar nuevo P de acuerdo a la posición elegida
    P[0] *= (1 + fac[I])
    P[1] *= (1 + fac[J])
    P[2] *= (1 + fac[K])
    P[3] *= (1 + fac[L])
    P[4] *= (1 + fac[M])

    # Verificar que la nueva solución P no viole los límites de factibilidad
    if Cumple_Limites(P, Lim):
        # Si es factible, calcular recompensa
        json_data = cargar_json(input_file)
        updated_data = update_json(json_data, *P)
        guardar_json(updated_data, "updated_data.json")

        w2_std = pyworld2.World2()
        w2_std.set_state_variables()
        w2_std.set_initial_state()
        w2_std.set_table_functions()
        w2_std.set_switch_functions("updated_data.json")
        w2_std.run()

        R_actual = w2_std.aveg_ql()  # Recompensa actual

        # Actualizar la matriz Q
        Q[I][J][K][L][M] += ((R_actual - R_previo) / R_previo)
        # Verificar si la nueva solución es mejor que la anterior
        if R_previo >= R_actual:
            P = P_previo  # Retornar a la solución previa
    else:
        # Penalizar soluciones no factibles
        Q[I][J][K][L][M] += -100
        P = P_previo  # Retornar a la solución previa

# Impresión de resultados finales
print("Parámetros finales =", P)

# Guardar datos finales
json_data = cargar_json(input_file)
updated_data = update_json(json_data, *P)
guardar_json(updated_data, "updated_data.json")

# Ejecutar el modelo final
w2_std = pyworld2.World2()
w2_std.set_state_variables()
w2_std.set_initial_state()
w2_std.set_table_functions()
w2_std.set_switch_functions("updated_data.json")
w2_std.run()

print("Recompensa final =", w2_std.aveg_ql())

# Generación de gráficos    
x = list(range(0, Runs + 1))  # Ejes x
plt.grid(True)
plt.title('Optimización de modelo DS con K-Armed Bandit algoritmo')
plt.xlabel('Iteraciones del método')
plt.ylabel('Valor variable objetivo')
plt.text(481, 50, f'Soluciones: inicial= {round(y[0], 2)}, final = {round(y[Runs - 1], 2)}')
plt.text(481, 45, f'% Ganancia = {round((y[Runs - 1] - y[0]) / y[0], 2)}')
plt.plot(x, y)  # Graficar puntos

# Gráfica del escenario optimizado
plot_world_variables(w2_std.time,
                     [w2_std.p, w2_std.polr, w2_std.ci, w2_std.ql, w2_std.nr],
                     ["P", "POLR", "CI", "QL", "NR"],
                     [[0, 8e9], [0, 40], [0, 20e9], [0, 2], [0, 1000e9]],
                     figsize=(7, 4), grid=True,
                     title="World2 - Optimized scenario")

plt.show()

