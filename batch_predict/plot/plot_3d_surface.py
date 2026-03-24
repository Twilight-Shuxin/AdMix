import sys
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

input_file = sys.argv[1]
output_file = f'{sys.argv[2]}.png' if len(sys.argv) > 2 else "plot_3d_surface.png"
output_file = os.path.join(os.path.dirname(__file__), 'outputs', os.path.basename(output_file))

df = pd.read_csv(input_file)

cached_bins = pd.cut(df["Cached Tkns"], bins=100)
sched_bins = pd.cut(df["Scheduled Tkns"], bins=100)
df["cached_bin"] = cached_bins.apply(lambda x: x.mid).astype(float)
df["sched_bin"] = sched_bins.apply(lambda x: x.mid).astype(float)

has_pred = "Pred Latency" in df.columns
if has_pred:
    df["z_abs"] = (df["Pred Latency"] - df["Latency"]).abs()
    df["z_rel"] = df["z_abs"] / df["Latency"]

    pivot_abs = df.groupby(["cached_bin", "sched_bin"])["z_abs"].mean().unstack()
    pivot_rel = df.groupby(["cached_bin", "sched_bin"])["z_rel"].mean().unstack()

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=["Absolute Error", "Relative Error"]
    )
    fig.add_trace(go.Surface(
        z=pivot_abs.values,
        x=pivot_abs.columns.tolist(),
        y=pivot_abs.index.tolist(),
        colorscale="Viridis",
        colorbar=dict(title="abs(pred - real)", x=0.45)
    ), row=1, col=1)
    fig.add_trace(go.Surface(
        z=pivot_rel.values,
        x=pivot_rel.columns.tolist(),
        y=pivot_rel.index.tolist(),
        colorscale="Viridis",
        colorbar=dict(title="abs(pred - real) / real", x=1.0)
    ), row=1, col=2)
    fig.update_layout(
        title="Interactive 3D Surface: Prediction Error vs Token Dims",
        scene=dict(xaxis_title="Scheduled Tkns", yaxis_title="Cached Tkns", zaxis_title="Absolute Error"),
        scene2=dict(xaxis_title="Scheduled Tkns", yaxis_title="Cached Tkns", zaxis_title="Relative Error"),
    )
else:
    df["z_val"] = df["Latency"]
    pivot = df.groupby(["cached_bin", "sched_bin"])["z_val"].mean().unstack()
    fig = go.Figure(data=[go.Surface(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="Viridis",
        colorbar=dict(title="Latency (ms)")
    )])
    fig.update_layout(
        title="Interactive 3D Surface: Latency vs Token Dims",
        scene=dict(xaxis_title="Scheduled Tkns", yaxis_title="Cached Tkns", zaxis_title="Latency")
    )

fig.write_image(output_file)
