import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CARGAR EL ARCHIVO
archivo = "Experimento_LotkaVolterra.csv"  
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
    "upper": upper, 
    "lower": lower
}).reset_index()

plt.figure(figsize=(10,6))

combinaciones = df_summary.groupby(["Nivel E", "Nivel fac"])

for (nivel_e, nivel_fac), data in combinaciones: 
    plt.plot(data["Corrida"], data["mean_y"], label=f"E={nivel_e}, Fac={nivel_fac}")
    plt.fill_between(data["Corrida"], data["lower"], data["upper"], alpha=0.2, color=color)

plt.xlabel("Corrida")
plt.ylabel("Retorno promedio (y)")
plt.title("GRÁFICO RENDIMIENTO IC 95%")
max_corrida = df_summary["Corrida"].max()
num_xticks = int(max_corrida/100)
plt.xticks(np.linspace(1, max_corrida, num=num_xticks, dtype=int))
num_yticks = 10 
y_min, y_max = df_summary["mean_y"].min(), df_summary["mean_y"].max()
plt.yticks(np.linspace(y_min, y_max, num=num_yticks))
plt.legend(title="Exploración, Paso", loc="upper left", fontsize=10, frameon=True, framealpha=0.3)
plt.grid(True)
plt.show()