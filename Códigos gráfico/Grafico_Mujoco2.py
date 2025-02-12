import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CARGAR EL ARCHIVO
archivo = "Experimento_World2.csv"
df = pd.read_csv(archivo, delimiter=",")

# AGRUPAR DATOS Y CALCULAR LOS DATOS PARA EL IC
df_agrupado = df.groupby(["Nivel E", "Nivel fac", "Corrida"])["y"]
mean_y = df_agrupado.mean()
std_y = df_agrupado.std()
n = df_agrupado.count()

# CALCULO DEL IC 95%

# Compute percentiles
lower = df_agrupado.quantile([0.025])
upper = df_agrupado.quantile([0.975])


# GRÁFICO PRINCIPAL
df_summary = pd.DataFrame({
    "mean_y": mean_y,
    "ic_95": ic_95,
    "upper": upper, 
    "lower": lower
}).reset_index()

plt.figure(figsize=(10,6))

combinaciones = df_summary.groupby(["Nivel E", "Nivel fac"])
colores_disponibles = plt.rcParams['axes.prop_cycle'].by_key()['color']  # Lista de colores de Matplotlib
colores = {}  # Diccionario para almacenar colores de cada serie

for (idx, ((nivel_e, nivel_fac), data)) in enumerate(combinaciones):
    color = colores_disponibles[idx % len(colores_disponibles)]  # Selecciona un color cíclicamente
    colores[(nivel_e, nivel_fac)] = color  # Guarda el color para usarlo después
    plt.plot(data["Corrida"], data["mean_y"], label=f"E={nivel_e}, Fac={nivel_fac}", color=color)
    plt.fill_between(data["Corrida"], data["lower"], data["upper"], alpha=0.2, color=color)
    

plt.xlabel("Corrida")
plt.ylabel("Retorno promedio (y)")
plt.title("GRÁFICO RENDIMIENTO IC 95%")
max_corrida = df_summary["Corrida"].max()
num_xticks = int(max_corrida / 100)
plt.xticks(np.linspace(1, max_corrida, num=num_xticks, dtype=int))
num_yticks = 10
y_min, y_max = df_summary["mean_y"].min(), df_summary["mean_y"].max()
plt.yticks(np.linspace(y_min, y_max, num=num_yticks))
plt.legend(title="Exploración, Paso", loc="upper left", fontsize=10, frameon=True, framealpha=0.3)
plt.grid(True)
plt.show()

# GRÁFICOS INDIVIDUALES
for (nivel_e, nivel_fac), data in combinaciones:
    plt.figure(figsize=(8, 5))
    color = colores[(nivel_e, nivel_fac)]  # Usar el mismo color del gráfico principal
    plt.plot(data["Corrida"], data["mean_y"], label=f"E={nivel_e}, Fac={nivel_fac}", color=color)
    plt.fill_between(data["Corrida"], data["mean_y"] - data["ic_95"],
                     data["mean_y"] + data["ic_95"], alpha=0.2, color=color)

    # Personalizar cada gráfico
    plt.xlabel("Corrida")
    plt.ylabel("Retorno promedio (y)")
    plt.title(f"Evolución de Retorno - E={nivel_e}, Fac={nivel_fac}")
    plt.xticks(np.linspace(1, max_corrida, num=num_xticks, dtype=int))
    plt.yticks(np.linspace(y_min, y_max, num=num_yticks))
    plt.legend(loc="upper left", fontsize=10, frameon=True, framealpha=0.3)
    plt.grid(True)

    # Mostrar el gráfico individual
    plt.show()
