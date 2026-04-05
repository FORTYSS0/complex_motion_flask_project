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
    if length is None:
        length = 5
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


def add_vector_with_arrow(fig, start, vector, color, name, scale=1.0):
    """Добавляет вектор со стрелкой на 3D график Plotly."""
    end = start + vector * scale
    
    # Рисуем линию вектора
    fig.add_trace(go.Scatter3d(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        z=[start[2], end[2]],
        mode='lines',
        line=dict(color=color, width=4),
        name=name,
        showlegend=True
    ))
    
    # Добавляем конус (стрелку) в конце вектора
    # Нормализуем направление
    direction = vector / (np.linalg.norm(vector) + 1e-10)
    # Размер конуса = 10% от длины вектора
    cone_size = np.linalg.norm(vector * scale) * 0.15
    
    fig.add_trace(go.Cone(
        x=[end[0] - direction[0] * cone_size * 0.5],
        y=[end[1] - direction[1] * cone_size * 0.5],
        z=[end[2] - direction[2] * cone_size * 0.5],
        u=[direction[0]],
        v=[direction[1]],
        w=[direction[2]],
        colorscale=[[0, color], [1, color]],
        showscale=False,
        sizemode="scaled",
        sizeref=cone_size,
        name=name + " (стрелка)",
        showlegend=False
    ))


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
    """Интерактивный график скоростей со стрелками."""
    point = data['point']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]],
                               mode='markers', marker=dict(color='red', size=8), name='Point M'))
    
    # Добавляем векторы со стрелками
    add_vector_with_arrow(fig, point, data['V_rel'], 'blue', 'V_rel')
    add_vector_with_arrow(fig, point, data['V_rot'], 'green', 'V_rot')
    add_vector_with_arrow(fig, point, data['V_trans_post'], 'orange', 'V_trans_post')
    add_vector_with_arrow(fig, point, data['V_abs'], 'purple', 'V_abs')
    
    fig.update_layout(
        title='Векторы скоростей в точке M',
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode='auto'),
        legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.5),
        margin=dict(l=0, r=0, t=30, b=50)
    )
    return fig.to_json()


def generate_interactive_accelerations(data):
    """Интерактивный график ускорений со стрелками."""
    point = data['point']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]],
                               mode='markers', marker=dict(color='red', size=8), name='Point M'))
    
    # Добавляем векторы со стрелками
    add_vector_with_arrow(fig, point, data['a_rel'], 'blue', 'a_rel')
    add_vector_with_arrow(fig, point, data['a_centr'], 'green', 'a_centr')
    add_vector_with_arrow(fig, point, data['a_rot'], 'orange', 'a_rot')
    add_vector_with_arrow(fig, point, data['a_trans_post'], 'brown', 'a_trans_post')
    add_vector_with_arrow(fig, point, data['a_cor'], 'cyan', 'a_cor')
    add_vector_with_arrow(fig, point, data['a_abs'], 'purple', 'a_abs')
    
    fig.update_layout(
        title='Векторы ускорений в точке M',
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode='auto'),
        legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.5),
        margin=dict(l=0, r=0, t=30, b=50)
    )
    return fig.to_json()


def generate_interactive_trajectory_with_velocities(data):
    """Комбинированный график: траектория + скорости со стрелками."""
    x_t, y_t, z_t, _ = get_trajectory_points()
    point = data['point']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x_t, y=y_t, z=z_t, mode='lines', 
                               line=dict(color='blue', width=4), name='Траектория'))
    fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]],
                               mode='markers', marker=dict(color='red', size=8), name='M (t=1)'))
    
    # Добавляем векторы скоростей со стрелками
    add_vector_with_arrow(fig, point, data['V_rel'], 'blue', 'V_rel')
    add_vector_with_arrow(fig, point, data['V_rot'], 'green', 'V_rot')
    add_vector_with_arrow(fig, point, data['V_trans_post'], 'orange', 'V_trans_post')
    add_vector_with_arrow(fig, point, data['V_abs'], 'purple', 'V_abs')
    
    fig.update_layout(
        title='Траектория и векторы скоростей',
        scene=dict(xaxis_title="X'", yaxis_title="Y'", zaxis_title="Z'", aspectmode='auto'),
        legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.5),
        margin=dict(l=0, r=0, t=30, b=50)
    )
    return fig.to_json()


def generate_interactive_trajectory_with_accelerations(data):
    """Комбинированный график: траектория + ускорения со стрелками."""
    x_t, y_t, z_t, _ = get_trajectory_points()
    point = data['point']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x_t, y=y_t, z=z_t, mode='lines', 
                               line=dict(color='blue', width=4), name='Траектория'))
    fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]],
                               mode='markers', marker=dict(color='red', size=8), name='M (t=1)'))
    
    # Добавляем векторы ускорений со стрелками
    add_vector_with_arrow(fig, point, data['a_rel'], 'blue', 'a_rel')
    add_vector_with_arrow(fig, point, data['a_centr'], 'green', 'a_centr')
    add_vector_with_arrow(fig, point, data['a_rot'], 'orange', 'a_rot')
    add_vector_with_arrow(fig, point, data['a_trans_post'], 'brown', 'a_trans_post')
    add_vector_with_arrow(fig, point, data['a_cor'], 'cyan', 'a_cor')
    add_vector_with_arrow(fig, point, data['a_abs'], 'purple', 'a_abs')
    
    fig.update_layout(
        title='Траектория и векторы ускорений',
        scene=dict(xaxis_title="X'", yaxis_title="Y'", zaxis_title="Z'", aspectmode='auto'),
        legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.5),
        margin=dict(l=0, r=0, t=30, b=50)
    )
    return fig.to_json()
