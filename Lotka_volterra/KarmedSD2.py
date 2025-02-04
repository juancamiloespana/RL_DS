# Se importa el modelo de Lotka Volterra
from LotkaVolterra import LotkaVolterraModel
# Importar librería para la media
from tkinter import Y
# Se importa paquete matematico numpy
import numpy as np
# Importamos librería genera números aleatorios
import random
# Importamos la libreria de graficas
import matplotlib.pyplot as plt 

# Programación de funciones a usar

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

# Inicio del algoritmo K-Armed bandid para optimizar parámetros del modelo de dinámica de sistemas

E = 0.1 # Parametro de exploracion
# fac = [-0.1, -0.05, -0.01, 0, 0.01, 0.05, 0.1] # Posibles cambios que pueden tenre los parámetros 
# fac = [-0.05, -0.01, 0, 0.01, 0.05]
fac = [-0.001, 0, 0.001]
P = [0.002, 0.04, 0.1, 0.0025] # Valor inicial de los parametros
Lim = [[0.0015,0.0025],[0.03,0.05],[0.05,0.15],[0.002,0.003]] # Rango de factibilidad de los parámetros
A = len(fac) # Número de acciones posibles
Q = np.zeros((A,A,A,A)) # Definición de la matriz Q del método K-Armed Bandid
Dim = Q.ndim # Dimensiones
Runs = 1500 # Corridas

lotka_volterra_instance = LotkaVolterraModel(*P)
R_inicial = lotka_volterra_instance.simulate() # Solución inicial del modelo de dinámica de sistemas

# Impresión de parámetros iniciales y solución inicial
print("Parametros iniciales = ", P)
print("Recompensa inicial =", round(R_inicial,5))

# Creacion de vector para graficar el retorno
y = [0]
y[0] = R_inicial

ejecucion = Runs/10

# Proceso de explorar - explotar
for i in range(Runs):

    P_previo = list(P) # Guardo el valor de P previo
    lotka_volterra_instance = LotkaVolterraModel(*P_previo)
    R_previo = lotka_volterra_instance.simulate()

    if (i % ejecucion) == 0:
        print(f'{(i/Runs)*100} % de ejecutado, Recompensa = {R_previo}')
    
    # Se grafica las ganancias
    y.append(R_previo)

    if E <random.random():
        # Opcion explotar
        max_val = np.amax(Q) # Identifico el máximo valor
        result = np.where(Q == max_val) # Encuentro la posición del máximo valor
        I, J, K, L = [result[i][0] for i in range(Dim)] # Guardo la posición del máximo valor
        
    else:
        # Proceso de exploracion
        idxs = [0] * Dim
        I, J, K, L = [random.randint(0, A-1) for items in idxs] # Genero una posición aleatoria

    # Asigno un nuevo P de acuerdo a la posición elegida
    P[0] = P[0] * (1+fac[I])
    P[1] = P[1] * (1+fac[J])
    P[2] = P[2] * (1+fac[K])
    P[3] = P[3] * (1+fac[L])
    
    # Verifico que esta nueva solución P no viole los límites de factibilidad
    if Cumple_Limites(P,Lim) == True:
        # Si sí es una solución factible calculo su recompensa y guardo esta posición
        lotka_volterra_instance = LotkaVolterraModel(*P)
        R_actual = lotka_volterra_instance.simulate()

        Q[I][J][K][L] += ((R_actual-R_previo)/R_previo)
        # Verifico que la nueva solución sea mejor que la mejor solución anterior, sino, me quedo con la solución mejor.
        if R_previo >= R_actual:
            P = P_previo
    else:
        # A una solulción que no satisfaga la factibilidad le asigno la penalidad 
        Q[I][J][K][L] += -100
        P = P_previo
 
# Impresion de los resultados finales
print("Parametros finales = ", P)
lotka_volterra_instance = LotkaVolterraModel(*P)
print("Recompensa final =", lotka_volterra_instance.simulate())

#lotka_volterra_instance.plot_simulation()

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
# function to show the plot
plt.show()