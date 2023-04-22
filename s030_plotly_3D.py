import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Equation of ring cyclide
# see https://en.wikipedia.org/wiki/Dupin_cyclide

a, b, d = 1.32, 1.0, 0.8
c = a**2 - b**2
u, v = np.mgrid[0 : 2 * np.pi : 100j, 0 : 2 * np.pi : 100j]
x = (d * (c - a * np.cos(u) * np.cos(v)) + b**2 * np.cos(u)) / (
    a - c * np.cos(u) * np.cos(v)
)
y = b * np.sin(u) * (a - d * np.cos(v)) / (a - c * np.cos(u) * np.cos(v))
z = b * np.sin(v) * (c * np.cos(u) - d) / (a - c * np.cos(u) * np.cos(v))

fig = make_subplots(
    rows=1,
    cols=1,
    specs=[[{"is_3d": True}]],
    subplot_titles=[
        "Subplot title",
    ],
)

cmap = plt.get_cmap("tab10")
# colorscale = [[0, "rgb" + str(cmap(1)[0:3])], [1, "rgb" + str(cmap(2)[0:3])]]
colorscale = [
    [0, "rgb(1.0, 0.5, 0.05)"],
    [1, "rgb(0.17, 0.63, 0.17)"],
]


offset = 5
options = dict(
    surfacecolor=np.ones_like(x),
    showscale=False,
    colorscale=colorscale,
)

fig.add_trace(
    go.Surface(
        x=x,
        y=y,
        z=z,
        colorbar_x=-0.07,
        **options,
    ),
    1,
    1,
)
fig.add_trace(
    go.Surface(
        x=x + offset,
        y=y,
        z=z,
        **options,
    ),
    1,
    1,
)
fig.update_layout(title_text="Title")
fig.show()
