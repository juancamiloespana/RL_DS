import pandas as pd
import numpy as np
import plotly.graph_objects as go
import colorcet as cc

def generate_learning_plots(df: pd.DataFrame, title="RL Agent Learning"):
    """
    Generate interactive learning curve visualization for reinforcement learning experiments.
    
    Parameters
    ----------
    df : pd.DataFrame
        Experimental results dataframe with required columns:
        - 'Epsilon_Level': Exploration probability (ε) for each treatment
        - 'Rho_Level': Parameter adjustment factor (ρ) - percentage modification
        - 'Run': Iteration number (x-axis)
        - 'Return': Performance metric value at each run
        
    title : str, optional
        Main title for the plot. Default is "RL Agent Learning".
    
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure object containing:
        - Mean learning curves for each treatment combination
        - Shaded confidence intervals (2.5th to 97.5th percentiles)
        - Hover tooltips with detailed information
        - Legend identifying each treatment by ε and ρ values
    """

    df_agrupado = df.groupby(["Epsilon_Level", "Rho_Level", "Run"])["Return"]
    mean_y = df_agrupado.mean()
    lower = df_agrupado.quantile(0.025)
    upper = df_agrupado.quantile(0.975)

    df_summary = pd.DataFrame({
        "mean_y": mean_y,
        "upper": upper,
        "lower": lower
    }).reset_index()
    
    grupos = sorted(
        df_summary.groupby(["Epsilon_Level", "Rho_Level"]).groups.keys(),
        key=lambda x: (x[1], x[0])
    )

    palette = cc.glasbey_light[:len(grupos)]
    rgba_colors = [f'rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.20)' for c in palette]
    line_colors = [f'rgb({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)})' for c in palette]

    fig = go.Figure()

    for j, (epsilon_level, rho_level) in enumerate(grupos):
        group_data = df_summary[
            (df_summary["Epsilon_Level"] == epsilon_level) & 
            (df_summary["Rho_Level"] == rho_level)
        ]
        group_name = f"ε = {epsilon_level}, ρ = {rho_level}"

        fill_color = rgba_colors[j % len(rgba_colors)]
        line_color = line_colors[j % len(line_colors)]

        # Confidence interval
        fig.add_trace(go.Scatter(
            x=group_data["Run"].tolist() + group_data["Run"].tolist()[::-1],
            y=group_data["upper"].tolist() + group_data["lower"].tolist()[::-1],
            fill='toself', 
            fillcolor=fill_color,
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            legendgroup=group_name,
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=group_data["Run"], 
            y=group_data["mean_y"],
            mode='lines', 
            name=group_name,
            line=dict(width=2.5, color=line_color),
            legendgroup=group_name,
            hovertemplate='<b>%{fullData.name}</b><br>Run: %{x}<br>Return: %{y:.2f}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=32, family="Segoe UI, Arial, sans-serif", color="#2c3e50")
        ),
        xaxis=dict(
            title="<i>Run</i>",
            title_font=dict(size=22, family="Segoe UI, Arial, sans-serif", color="#2c3e50"),
            tickfont=dict(size=18, family="Segoe UI, Arial, sans-serif", color="#34495e"),
            showgrid=True,
            gridwidth=0.8,
            gridcolor='rgba(211, 211, 211, 0.5)',
            zeroline=False,
            showline=False,
            mirror=False
        ),
        yaxis=dict(
            title="<i>Average Return (ȳ)</i>",
            title_font=dict(size=22, family="Segoe UI, Arial, sans-serif", color="#2c3e50"),
            tickfont=dict(size=18, family="Segoe UI, Arial, sans-serif", color="#34495e"),
            showgrid=True,
            gridwidth=0.8,
            gridcolor='rgba(211, 211, 211, 0.5)',
            zeroline=False,
            showline=False,
            mirror=False
        ),
        legend=dict(
            title=dict(
                text="<b>Treatments</b><br>",
                font=dict(size=20, family="Segoe UI, Arial, sans-serif", color="#2c3e50")
            ),
            title_side="top center",
            font=dict(size=18, family="Segoe UI, Arial, sans-serif", color="#34495e"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0)",
            borderwidth=0,
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            orientation="v"
        ),
        margin=dict(l=90, r=250, t=100, b=80),
        template="plotly_white",
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Segoe UI, Arial, sans-serif"
        )
    )

    fig.show()
    return fig