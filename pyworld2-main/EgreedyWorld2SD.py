# Carga de paquetes
import json
# Importar librería para la media
from tkinter import Y
# Se importa paquete matematico numpy
import math
import random
import numpy as np
# Carga de los paquetes del World2
import pyworld2
from pyworld2.utils import plot_world_variables, plt
import os
from tqdm import tqdm

# Funcion para cambiar el archivo JSON data para el World2
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

# Funcion del e-greedy

# Función para verificar si se violan los rangos de factibilidad de los parámetros a optimizar
def Cumple_Limites(p, l):
    
    suma = [0]*len(p)
    
    # Si algún parámetro se sale de los límites se devuelve verdadero
    for i in range(len(p)):
        if p[i] < l[i][0] or p[i] > l[i][1]:
            suma[i]=1
    
    if sum(suma) > 0:
        return False
    else:
        return True
    
# Codigo del e-greedy

E = 0.4 # Parametro de exploracion
fac = [-0.01, 0, 0.01] # Posibles cambios que pueden tenre los parámetros
P = [0.028, 0.25, 0.8, 0.03, 0.5] # Valor inicial de los parametros
Lim = [[0.02,0.04],[0.1,1.0],[0.6,1.25],[0.02,0.04],[0.1,1.0]] # Rango de factibilidad de los parámetros
# Estos rangos provienen del articulo Using White-box nonlinear 2017 por Vierhaus
A = len(fac) # Número de acciones posibles
Q = np.zeros((A,A,A,A,A)) # Definición de la matriz Q del método K-Armed Bandid
Dim = Q.ndim # Dimensiones
Runs = 500 # Corridas

input_file = os.path.join(os.path.dirname(__file__), "pyworld2", "functions_switch_default.json")

# Leer el archivo JSON y guardar los datos
with open(input_file, "r") as file:
    json_data = json.load(file)
    
# Cambiar los valores de los parametros en los datos
updated_data = update_json(json_data, P[0], P[1], P[2], P[3], P[4])

# Creacion de un nuevo archivo json con los datos cambiados
output_file = "updated_data.json"
with open(output_file, "w") as file:
    json.dump(updated_data, file, indent=4)

# Primera corrida del modelo para calcular la recompensa inicial
w2_std = pyworld2.World2()
w2_std.set_state_variables()
w2_std.set_initial_state()
w2_std.set_table_functions()
w2_std.set_switch_functions(output_file )
w2_std.run()

R_inicial = w2_std.aveg_ql() # Solución inicial del modelo de dinámica de sistemas

# Impresión de parámetros iniciales y solución inicial
print("Parametros iniciales = ", P)
print("Recompensa inicial =", round(R_inicial,5))
print("")

# Creacion de vector para graficar el retorno
y = [0]
y[0] = R_inicial

ejecucion = Runs/10

# Proceso de explorar - explotar
for i in tqdm(range(Runs)):

    P_previo = list(P) # Guardo el valor de P previo

    #print(f'el P_previo es {P_previo}')

    input_file = os.path.join(os.path.dirname(__file__), "pyworld2", "functions_switch_default.json")

    # Leer el archivo JSON y guardar los datos
    with open(input_file, "r") as file:
        json_data = json.load(file)
    
    # Cambiar los valores de los parametros en los datos
    updated_data = update_json(json_data, P[0], P[1], P[2], P[3], P[4])

    # Creacion de un nuevo archivo json con los datos cambiados
    output_file = "updated_data.json"
    with open(output_file, "w") as file:
        json.dump(updated_data, file, indent=4)

    w2_std = pyworld2.World2()
    w2_std.set_state_variables()
    w2_std.set_initial_state()
    w2_std.set_table_functions()
    w2_std.set_switch_functions(output_file)
    w2_std.run()

    R_previo = w2_std.aveg_ql()

    #print(f'La recompensa previa es {R_previo}')

    if (i % ejecucion) == 0:
       print(f'{(i/Runs)*100} % de ejecutado, Recompensa = {R_previo}')
  

    # Se grafica las ganancias
    y.append(R_previo)

    if E <random.random():
        # Opcion explotar
        max_val = np.amax(Q) # Identifico el máximo valor
        result = np.where(Q == max_val) # Encuentro la posición del máximo valor
        I, J, K, L, M = [result[i][0] for i in range(Dim)] # Guardo la posición del máximo valor
        
    else:
        # Proceso de exploracion
        idxs = [0] * Dim
        I, J, K, L, M = [random.randint(0, A-1) for items in idxs] # Genero una posición aleatoria
    
    # Asigno un nuevo P de acuerdo a la posición elegida
    P[0] = P[0] * (1+fac[I])
    P[1] = P[1] * (1+fac[J])
    P[2] = P[2] * (1+fac[K])
    P[3] = P[3] * (1+fac[L])
    P[4] = P[4] * (1+fac[M])

    # Verifico que esta nueva solución P no viole los límites de factibilidad
    if Cumple_Limites(P,Lim) == True:
        # Si sí es una solución factible calculo su recompensa y guardo esta posición

        input_file = os.path.join(os.path.dirname(__file__), "pyworld2", "functions_switch_default.json")

    # Leer el archivo JSON y guardar los datos
        with open(input_file, "r") as file:
            json_data = json.load(file)
    
    # Cambiar los valores de los parametros en los datos
        updated_data = update_json(json_data, P[0], P[1], P[2], P[3], P[4])

    # Creacion de un nuevo archivo json con los datos cambiados
        output_file = "updated_data.json"
        with open(output_file, "w") as file:
            json.dump(updated_data, file, indent=4)

        w2_std = pyworld2.World2()
        w2_std.set_state_variables()
        w2_std.set_initial_state()
        w2_std.set_table_functions()
        w2_std.set_switch_functions(output_file)
        w2_std.run()
        
        R_actual = w2_std.aveg_ql()

        #print(f'Los valores P de la corrida {i} son {P}')
        #print(f'La recompensa de la corrida {i} es {R_actual}')
        #print("")

        Q[I][J][K][L][M] += ((R_actual-R_previo)/R_previo)
        # Verifico que la nueva solución sea mejor que la mejor solución anterior, sino, me quedo con la solución mejor.
        if R_previo >= R_actual:
            P = P_previo
    else:
        # A una solulción que no satisfaga la factibilidad le asigno la penalidad 
        Q[I][J][K][L][M] += -100
        P = P_previo

# Impresion de los resultados finales
print("")
print("Parametros finales = ", P)

input_file = os.path.join(os.path.dirname(__file__), "pyworld2", "functions_switch_default.json")

    # Leer el archivo JSON y guardar los datos
with open(input_file, "r") as file:
    json_data = json.load(file)
    
# Cambiar los valores de los parametros en los datos
updated_data = update_json(json_data, P[0], P[1], P[2], P[3], P[4])

# Creacion de un nuevo archivo json con los datos cambiados
output_file = "updated_data.json"
with open(output_file, "w") as file:
    json.dump(updated_data, file, indent=4)

w2_std = pyworld2.World2()
w2_std.set_state_variables()
w2_std.set_initial_state()
w2_std.set_table_functions()
w2_std.set_switch_functions(output_file)
w2_std.run()
        
print("Recompensa final =", w2_std.aveg_ql())


# Generacion de graficos    
# x axis values
x = list(range(0,Runs+1))
# Grilla y ejes
plt.grid(True)
plt.title('Optimización de modelo DS con K-Armed Bandid algoritmo')
plt.xlabel('Iteraciones del método')
plt.ylabel('Valor variable objetivo')
plt.text(481, 50, f'Soluciones: inicial= {round(y[0],2)}, final = {round(y[Runs-1],2)}')
plt.text(481, 45, f'% Ganancia = {round((y[Runs-1]-y[0])/y[0], 2)}')
# plotting the points 
plt.plot(x, y)

# Grafica del escenario optimizado

plot_world_variables(w2_std.time,
                     [w2_std.p, w2_std.polr, w2_std.ci, w2_std.ql, w2_std.nr],
                      ["P", "POLR", "CI", "QL", "NR"],
                      [[0, 8e9], [0, 40], [0, 20e9], [0, 2], [0, 1000e9]],
                      figsize=(7, 4), grid=True,
                      title="World2 - Optimized scenario")

plt.show()