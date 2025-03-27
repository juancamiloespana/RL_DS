import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Cargar CSV
archivo = "Experimento_LotkaVolterra50.csv"
df = pd.read_csv(archivo, delimiter=",")

# Agrupar data y calcular
df_agrupado = df.groupby(["Nivel E", "Nivel fac", "Corrida"])["y"]
mean_y = df_agrupado.mean() 
lower = df_agrupado.quantile(0.025) 
upper = df_agrupado.quantile(0.975) 

df_summary = pd.DataFrame({
    "mean_y": mean_y,
    "upper": upper,
    "lower": lower
}).reset_index()

# CREAR EL GRÁFICO
# Colores
mpl_colors = plt.get_cmap("tab10").colors + plt.get_cmap("Set3").colors  # 10 + 12 colors
rgba_colors = [f'rgba({int(r*255)},{int(g*255)},{int(b*255)},0.15)' for r, g, b in mpl_colors]
line_colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in mpl_colors]

fig = go.Figure()
color_index = 0

# Series
for (nivel_e, nivel_fac), group_data in df_summary.groupby(["Nivel E", "Nivel fac"]):
    group_name = f"ϵ={nivel_e}, α={nivel_fac}"
    color = rgba_colors[color_index % len(rgba_colors)]
    line_color = line_colors[color_index % len(line_colors)]
    color_index += 1

    # CIs
    fig.add_trace(go.Scatter(
        x=group_data["Corrida"].tolist() + group_data["Corrida"].tolist()[::-1],
        y=group_data["upper"].tolist() + group_data["lower"].tolist()[::-1],
        fill='toself', fillcolor=color,
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        legendgroup=group_name
    ))

    # Linea principal de las series
    fig.add_trace(go.Scatter(
        x=group_data["Corrida"], y=group_data["mean_y"],
        mode='lines', name=group_name,
        line=dict(width=2, color=line_color),
        legendgroup=group_name
    ))

# Estética
fig.update_layout(
    title="<b>Evolution of average return based on ϵ and α</b>",  
    title_x=0.5,
    title_font=dict(size=24),
    xaxis=dict(
        title="Run",
        title_font=dict(size=16),
        tickfont=dict(size=12) 
    ),
    yaxis=dict(
        title="Average return (y)",
        title_font=dict(size=16),
        tickfont=dict(size=12),
        dtick=5
    ),
    legend_title_font=dict(size=18),
    legend=dict(
        font=dict(size=16),
        title_side="top",
        yanchor="top", y=1,
        xanchor="left", x=1.02,
        orientation="v"
    ),
    template="plotly_white"
)

fig.show()
