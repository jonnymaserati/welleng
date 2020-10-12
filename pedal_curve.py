import numpy as np
import plotly.graph_objects as go

# ellipse
a = 3
b = 2
c = 0
u = np.linspace(0,2*np.pi,100)
v = np.linspace(0,np.pi,100)
u, v = np.meshgrid(u,v)
u = u.flatten()
v = v.flatten()
v = np.full_like(v, np.pi/2)

x = a * np.cos(u) * np.sin(v)
y = b * np.sin(u) * np.sin(v)
z = c * np.cos(v)

fig = go.Figure(
    data = go.Scatter3d(
        x = x,
        y = y,
        z = z
    )
)

v = np.zeros_like(v)

e = a * (
    (b ** 2 * np.cos(u))
    / 
    (b ** 2 * (np.cos(u)) ** 2 + a ** 2 * (np.sin(u)) ** 2)
)

f = b * (
    (a ** 2 * np.sin(u))
    /
    (b ** 2 * (np.cos(u)) ** 2 + a ** 2 * (np.sin(u)) ** 2)
)

g = np.zeros_like(e)

R = np.sqrt(e ** 2 + f ** 2)

fig.add_trace(
    go.Scatter3d(
        x = e,
        y = f,
        z = g
    )
)

def get_PCR(a, b, t):
    f = a * (
        (b ** 2 * np.cos(t))
        / 
        (b ** 2 * (np.cos(t)) ** 2 + a ** 2 * (np.sin(t)) ** 2)
    )

    g = b * (
        (a ** 2 * np.sin(t))
        /
        (b ** 2 * (np.cos(t)) ** 2 + a ** 2 * (np.sin(t)) ** 2)
    )

    R = np.sqrt(f ** 2 + g ** 2)

fig.show(renderer="iframe")