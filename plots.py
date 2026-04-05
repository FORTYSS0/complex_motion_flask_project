import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch
import plotly.graph_objects as go

# ==================== Вспомогательные классы и функции

class Arrow3D(FancyArrowPatch):
    """3D стрелка для matplotlib."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def draw_axes(ax, origin=(0,0,0), length=None, color='black', labels=['X', 'Y', 'Z']):
    """Рисует оси координат в 3D."""
    for i, label in enumerate(labels):
        vec = [0,0,0]
        vec[i] = length
        ax.quiver(origin[0], origin[1], origin[2],
                  vec[0], vec[1], vec[2],
                  color=color, label=label, arrow_length_ratio=0.03, linewidth=2)
        ax.text(origin[0]+vec[0], origin[1]+vec[1], origin[2]+vec[2], label,
                color=color, fontsize=10, ha='center', va='center')


def get_trajectory_points(t_max=2.5, num_points=200):
    """Возвращает точки траектории для построения."""
    t_vals = np.linspace(0, t_max, num_points)
    x_t = 8 * np.cos(np.pi * t_vals**2 / 3)
    y_t = 16 * np.sin(np.pi * t_vals**2 / 3)
    z_t = 2 * t_vals**2 + 4
    return x_t, y_t, z_t, t_vals


# ==================== Интерактивные графики (Plotly) 

def generate_interactive_trajectory(data):
    """Возвращает JSON для интерактивного графика траектории."""
    x_t, y_t, z_t, _ = get_trajectory_points()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x_t, y=y_t, z=z_t, mode='lines', 
                               line=dict(color='blue', width=4), name='Траектория'))
    
    # Точка M
    fig.add_trace(go.Scatter3d(x=[data['point'][0]], y=[data['point'][1]], z=[data['point'][2]],
                               mode='markers', marker=dict(color='red', size=8), name='M (t=1)'))
    
    fig.update_layout(
        title='Способ задания движения (абсолютная траектория)',
        scene=dict(xaxis_title="X'", yaxis_title="Y'", zaxis_title="Z'", aspectmode='auto'),
        legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.5),
        margin=dict(l=0, r=0, t=30, b=50)
    )
    return fig.to_json()


def generate_interactive_velocities(data):
    """Интерактивный график скоростей."""
    point = data['point']
    vectors = [data['V_rel'], data['V_rot'], data['V_trans_post'], data['V_abs']]
    colors = ['blue', 'green', 'orange', 'purple']
    labels = ['V_rel', 'V_rot', 'V_trans_post', 'V_abs']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]],
                               mode='markers', marker=dict(color='red', size=8), name='Point M'))
    
    for vec, col, lab in zip(vectors, colors, labels):
        fig.add_trace(go.Scatter3d(x=[point[0], point[0]+vec[0]],
                                   y=[point[1], point[1]+vec[1]],
                                   z=[point[2], point[2]+vec[2]],
                                   mode='lines', line=dict(color=col, width=3), name=lab))
    
    fig.update_layout(
        title='Векторы скоростей в точке M',
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode='auto'),
        legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.5),
        margin=dict(l=0, r=0, t=30, b=50)
    )
    return fig.to_json()


def generate_interactive_accelerations(data):
    """Интерактивный график ускорений."""
    point = data['point']
    vectors = [data['a_rel'], data['a_centr'], data['a_rot'], data['a_trans_post'], data['a_cor'], data['a_abs']]
    colors = ['blue', 'green', 'orange', 'brown', 'cyan', 'purple']
    labels = ['a_rel', 'a_centr', 'a_rot', 'a_trans_post', 'a_cor', 'a_abs']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]],
                               mode='markers', marker=dict(color='red', size=8), name='Point M'))
    
    for vec, col, lab in zip(vectors, colors, labels):
        fig.add_trace(go.Scatter3d(x=[point[0], point[0]+vec[0]],
                                   y=[point[1], point[1]+vec[1]],
                                   z=[point[2], point[2]+vec[2]],
                                   mode='lines', line=dict(color=col, width=3), name=lab))
    
    fig.update_layout(
        title='Векторы ускорений в точке M',
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode='auto'),
        legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.5),
        margin=dict(l=0, r=0, t=30, b=50)
    )
    return fig.to_json()


def generate_interactive_trajectory_with_velocities(data):
    """Комбинированный интерактивный график: траектория + скорости."""
    x_t, y_t, z_t, _ = get_trajectory_points()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x_t, y=y_t, z=z_t, mode='lines', 
                               line=dict(color='blue', width=4), name='Траектория'))
    fig.add_trace(go.Scatter3d(x=[data['point'][0]], y=[data['point'][1]], z=[data['point'][2]],
                               mode='markers', marker=dict(color='red', size=8), name='M (t=1)'))
    
    # Векторы скоростей (масштабированные)
    all_points = np.vstack([np.column_stack([x_t, y_t, z_t]), [data['point']], [0,0,0]])
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    axis_len = np.max(max_vals - min_vals) * 0.3
    
    vectors = [data['V_rel'], data['V_rot'], data['V_trans_post'], data['V_abs']]
    colors = ['blue', 'green', 'orange', 'purple']
    labels = ['V_rel', 'V_rot', 'V_trans_post', 'V_abs']
    
    for vec, col, lab in zip(vectors, colors, labels):
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            scaled_vec = (vec / norm) * axis_len
            fig.add_trace(go.Scatter3d(x=[data['point'][0], data['point'][0]+scaled_vec[0]],
                                       y=[data['point'][1], data['point'][1]+scaled_vec[1]],
                                       z=[data['point'][2], data['point'][2]+scaled_vec[2]],
                                       mode='lines', line=dict(color=col, width=3), name=lab))
    
    fig.update_layout(
        title='Траектория и векторы скоростей',
        scene=dict(xaxis_title="X'", yaxis_title="Y'", zaxis_title="Z'", aspectmode='auto'),
        legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.5),
        margin=dict(l=0, r=0, t=30, b=50)
    )
    return fig.to_json()


def generate_interactive_trajectory_with_accelerations(data):
    """Комбинированный интерактивный график: траектория + ускорения."""
    x_t, y_t, z_t, _ = get_trajectory_points()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x_t, y=y_t, z=z_t, mode='lines', 
                               line=dict(color='blue', width=4), name='Траектория'))
    fig.add_trace(go.Scatter3d(x=[data['point'][0]], y=[data['point'][1]], z=[data['point'][2]],
                               mode='markers', marker=dict(color='red', size=8), name='M (t=1)'))
    
    # Векторы ускорений (масштабированные)
    all_points = np.vstack([np.column_stack([x_t, y_t, z_t]), [data['point']], [0,0,0]])
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    axis_len = np.max(max_vals - min_vals) * 0.3
    
    vectors = [data['a_rel'], data['a_centr'], data['a_rot'], data['a_trans_post'], data['a_cor'], data['a_abs']]
    colors = ['blue', 'green', 'orange', 'brown', 'cyan', 'purple']
    labels = ['a_rel', 'a_centr', 'a_rot', 'a_trans_post', 'a_cor', 'a_abs']
    
    for vec, col, lab in zip(vectors, colors, labels):
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            scaled_vec = (vec / norm) * axis_len
            fig.add_trace(go.Scatter3d(x=[data['point'][0], data['point'][0]+scaled_vec[0]],
                                       y=[data['point'][1], data['point'][1]+scaled_vec[1]],
                                       z=[data['point'][2], data['point'][2]+scaled_vec[2]],
                                       mode='lines', line=dict(color=col, width=3), name=lab))
    
    fig.update_layout(
        title='Траектория и векторы ускорений',
        scene=dict(xaxis_title="X'", yaxis_title="Y'", zaxis_title="Z'", aspectmode='auto'),
        legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.5),
        margin=dict(l=0, r=0, t=30, b=50)
    )
    return fig.to_json()
