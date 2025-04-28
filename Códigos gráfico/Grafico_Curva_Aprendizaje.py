import pandas as pd
import numpy as np
import plotly.graph_objects as go
import colorcet as cc
import math

def plots(df: pd.DataFrame, max_lines_per_plot=4, title_prefix="Average return"):
    
    df_agrupado = df.groupby(["Nivel E", "Nivel fac", "Corrida"])["y"]
    mean_y = df_agrupado.mean()
    lower = df_agrupado.quantile(0.025)
    upper = df_agrupado.quantile(0.975)

    df_summary = pd.DataFrame({
        "mean_y": mean_y,
        "upper": upper,
        "lower": lower
    }).reset_index()

    # Tratamientos
    grupos = sorted(
        df_summary.groupby(["Nivel E", "Nivel fac"]).groups.keys(),
        key=lambda x: (x[1], x[0])  # Ordenar por rho, luego epsilon
    )

    total_plots = math.ceil(len(grupos) / max_lines_per_plot)

    # Colores
    palette = cc.glasbey_light[:max_lines_per_plot * total_plots]
    rgba_colors = [f'rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.15)' for c in palette]
    line_colors = [f'rgb({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)})' for c in palette]

    for i in range(total_plots):
        fig = go.Figure()
        subset = grupos[i * max_lines_per_plot:(i + 1) * max_lines_per_plot]

        for j, (nivel_e, nivel_fac) in enumerate(subset):
            group_data = df_summary[(df_summary["Nivel E"] == nivel_e) & (df_summary["Nivel fac"] == nivel_fac)]
            group_name = f"ϵ={nivel_e}, ρ={nivel_fac}"

            color_idx = i * max_lines_per_plot + j
            fill_color = rgba_colors[color_idx % len(rgba_colors)]
            line_color = line_colors[color_idx % len(line_colors)]

            # CIs
            fig.add_trace(go.Scatter(
                x=group_data["Corrida"].tolist() + group_data["Corrida"].tolist()[::-1],
                y=group_data["upper"].tolist() + group_data["lower"].tolist()[::-1],
                fill='toself', fillcolor=fill_color,
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                legendgroup=group_name
            ))

            # Línea principal
            fig.add_trace(go.Scatter(
                x=group_data["Corrida"], y=group_data["mean_y"],
                mode='lines', name=group_name,
                line=dict(width=2, color=line_color),
                legendgroup=group_name
            ))

        # Estética
        fig.update_layout(
            title=f"<b>{title_prefix} — Subplot {i+1}</b>",
            title_x=0.5,
            title_font=dict(size=26),
            xaxis=dict(
                title="Run",
                title_font=dict(size=24),
                tickfont=dict(size=24)
            ),
            yaxis=dict(
                title="Average return (y)",
                title_font=dict(size=24),
                tickfont=dict(size=24),
                #dtick=5
            ),
            legend_title_font=dict(size=20),
            legend=dict(
                font=dict(size=24),
                title_side="top",
                yanchor="top", y=1,
                xanchor="left", x=1.02,
                orientation="v"
            ),
            template="plotly_white"
        )

        fig.show()

# Cargar datos y ejecutar
df = pd.read_csv("Experimento_world2.csv", delimiter=",")
plots(df, title_prefix="RL Agent Learning (World2)")
