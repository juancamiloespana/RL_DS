import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
archivo = "Experimento_LotkaVolterra.csv"
df = pd.read_csv(archivo, delimiter=",")

# Group data by 'Nivel E', 'Nivel fac', and 'Corrida' and calculate mean, std, and count
df_agrupado = df.groupby(["Nivel E", "Nivel fac", "Corrida"])["y"]
mean_y = df_agrupado.mean()  # Calculate mean for each group
lower = df_agrupado.quantile(0.025)  # Calculate the 2.5th percentile
upper = df_agrupado.quantile(0.975)  # Calculate the 97.5th percentile

# Create a summary DataFrame with mean and confidence intervals
df_summary = pd.DataFrame({
    "mean_y": mean_y,
    "upper": upper,
    "lower": lower
}).reset_index()

# Define colors from Matplotlib
mpl_colors = plt.get_cmap("tab10").colors + plt.get_cmap("Set3").colors  # 10 + 12 colors
rgba_colors = [f'rgba({int(r255)},{int(g255)},{int(b*255)},0.15)' for r, g, b in mpl_colors]
line_colors = [f'rgb({int(r255)},{int(g255)},{int(b*255)})' for r, g, b in mpl_colors]

# Create an interactive plot using Plotly with confidence intervals
fig = go.Figure()
color_index = 0

# Add traces for each group with confidence intervals
for (nivel_e, nivel_fac), group_data in df_summary.groupby(["Nivel E", "Nivel fac"]):
    group_name = f"E={nivel_e}, Fac={nivel_fac}"
    color = rgba_colors[color_index % len(rgba_colors)]
    line_color = line_colors[color_index % len(line_colors)]
    color_index += 1

    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=group_data["Corrida"].tolist() + group_data["Corrida"].tolist()[::-1],
        y=group_data["upper"].tolist() + group_data["lower"].tolist()[::-1],
        fill='toself', fillcolor=color,
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        legendgroup=group_name
    ))

    # Add the main line plot
    fig.add_trace(go.Scatter(
        x=group_data["Corrida"], y=group_data["mean_y"],
        mode='lines', name=group_name,
        line=dict(width=2, color=line_color),
        legendgroup=group_name
    ))

# Customize the layout
fig.update_layout(
    title="<b>Chart</b>",
    title_x=0.5,
    xaxis_title="Run",
    yaxis_title="Average Return (y)",
    legend_title="Exploration and Fac",
    template="plotly_white"
)

# Show the interactive plot
fig.show()
