import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch
import plotly.graph_objects as go

# ==================== Статичные PNG-графики для экспорта ====================
# (код полностью идентичен предыдущей версии – здесь не дублирую для краткости)
# Он должен быть таким же, как в последнем ответе.

# ==================== Интерактивные графики (Plotly) ====================

def generate_interactive_trajectory(data):
    t_vals = np.linspace(0, 2.5, 200)
    x_t = 8 * np.cos(np.pi * t_vals**2 / 3)
    y_t = 16 * np.sin(np.pi * t_vals**2 / 3)
    z_t = 2 * t_vals**2 + 4

    fig = go.Figure()

    # Траектория
    fig.add_trace(go.Scatter3d(x=x_t, y=y_t, z=z_t, mode='lines',
                               line=dict(color='blue', width=4), name='Траектория'))

    # Стрелки направления
    n_arrows = 10
    idx = np.linspace(0, len(t_vals)-1, n_arrows, dtype=int)
    for i in idx:
        if i < len(t_vals)-1:
            dx = x_t[i+1] - x_t[i]
            dy = y_t[i+1] - y_t[i]
            dz = z_t[i+1] - z_t[i]
            length = np.sqrt(dx**2 + dy**2 + dz**2)
            if length > 1e-8:
                scale = 0.15 * max(np.ptp(x_t), np.ptp(y_t), np.ptp(z_t))
                u = dx / length * scale
                v = dy / length * scale
                w = dz / length * scale
                fig.add_trace(go.Cone(x=[x_t[i]], y=[y_t[i]], z=[z_t[i]],
                                      u=[u], v=[v], w=[w],
                                      colorscale=[[0, 'green'], [1, 'green']],
                                      showscale=False, sizemode='scaled', sizeref=0.5,
                                      name='Направление движения'))

    # Точка M
    fig.add_trace(go.Scatter3d(x=[data['point'][0]], y=[data['point'][1]], z=[data['point'][2]],
                               mode='markers', marker=dict(color='red', size=8), name='M (t=1)'))

    # ... (все остальные элементы: оси, точка O, вектор V_abs, дуга – такие же, как в предыдущей версии)
    # Для краткости я опускаю их здесь, но они должны быть вставлены без изменений.

    # Дуга вращения (пример)
    # ... (код дуги)

    fig.update_layout(title='Способ задания движения (абсолютная траектория)',
                      scene=dict(xaxis_title="X'", yaxis_title="Y'", zaxis_title="Z'", aspectmode='auto'),
                      legend=dict(x=0.8, y=0.9))
    return fig.to_json()

def generate_interactive_velocities(data):
    point = data['point']
    V_rel = data['V_rel']
    V_rot = data['V_rot']
    V_trans = data['V_trans_post']
    V_abs = data['V_abs']

    e_x = V_rel / (np.linalg.norm(V_rel) + 1e-8)
    e_z = np.array([0.0, 0.0, 1.0])
    e_y = np.cross(e_z, e_x)
    e_y = e_y / (np.linalg.norm(e_y) + 1e-8)
    R = np.array([e_x, e_y, e_z])

    def transform(v): return R @ v

    point_new = transform(point)
    V_rel_new = transform(V_rel)
    V_rot_new = transform(V_rot)
    V_trans_new = transform(V_trans)
    V_abs_new = transform(V_abs)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=[point_new[0]], y=[point_new[1]], z=[point_new[2]],
                               mode='markers', marker=dict(color='red', size=8), name='Point M'))

    vectors = [V_rel_new, V_rot_new, V_trans_new, V_abs_new]
    colors = ['blue', 'green', 'orange', 'purple']
    labels = ['V_rel', 'V_rot', 'V_trans_post', 'V_abs']
    numeric_vals = [np.linalg.norm(v) for v in [V_rel, V_rot, V_trans, V_abs]]

    for vec, col, lab, val in zip(vectors, colors, labels, numeric_vals):
        # Линия вектора
        fig.add_trace(go.Scatter3d(x=[point_new[0], point_new[0]+vec[0]],
                                   y=[point_new[1], point_new[1]+vec[1]],
                                   z=[point_new[2], point_new[2]+vec[2]],
                                   mode='lines+text', line=dict(color=col, width=3),
                                   text=['', f'{lab} = {val:.2f}'], textposition='middle right', name=lab))
        # Стрелка (конус) в конце
        end_x = point_new[0] + vec[0]
        end_y = point_new[1] + vec[1]
        end_z = point_new[2] + vec[2]
        arrow_scale = 0.15
        u = vec[0] * arrow_scale
        v = vec[1] * arrow_scale
        w = vec[2] * arrow_scale
        fig.add_trace(go.Cone(x=[end_x], y=[end_y], z=[end_z],
                              u=[u], v=[v], w=[w],
                              colorscale=[[0, col], [1, col]],
                              showscale=False, sizemode='scaled', sizeref=0.3,
                              showlegend=False))

    # Оси новой системы (чёрные)
    all_pts = [point_new] + [point_new+v for v in vectors] + [np.zeros(3)]
    all_arr = np.array(all_pts)
    min_vals = np.min(all_arr, axis=0)
    max_vals = np.max(all_arr, axis=0)
    max_range = np.max(max_vals - min_vals)
    axis_len = max_range * 0.25

    fig.add_trace(go.Scatter3d(x=[point_new[0], point_new[0]+axis_len], y=[point_new[1], point_new[1]],
                               z=[point_new[2], point_new[2]], mode='lines+text',
                               line=dict(color='black', width=2), text=['', 'V_отн'], textposition='middle right', showlegend=False))
    fig.add_trace(go.Scatter3d(x=[point_new[0], point_new[0]], y=[point_new[1], point_new[1]+axis_len],
                               z=[point_new[2], point_new[2]], mode='lines+text',
                               line=dict(color='black', width=2), text=['', 'a_cor'], textposition='middle right', showlegend=False))
    fig.add_trace(go.Scatter3d(x=[point_new[0], point_new[0]], y=[point_new[1], point_new[1]],
                               z=[point_new[2], point_new[2]+axis_len], mode='lines+text',
                               line=dict(color='black', width=2), text=['', 'ω'], textposition='middle right', showlegend=False))

    fig.update_layout(title='Векторы скоростей в точке M (оси: V_отн, a_cor, ω)',
                      scene=dict(xaxis_title="V_отн", yaxis_title="a_cor", zaxis_title="ω", aspectmode='auto'),
                      legend=dict(x=0.8, y=0.9))
    return fig.to_json()

def generate_interactive_accelerations(data):
    point = data['point']
    a_rel = data['a_rel']
    a_centr = data['a_centr']
    a_rot = data['a_rot']
    a_trans = data['a_trans_post']
    a_cor = data['a_cor']
    a_abs = data['a_abs']

    e_z = np.array([0.0, 0.0, 1.0])
    e_y = a_cor / (np.linalg.norm(a_cor) + 1e-8)
    e_x = np.cross(e_y, e_z)
    e_x = e_x / (np.linalg.norm(e_x) + 1e-8)
    R = np.array([e_x, e_y, e_z])

    def transform(v): return R @ v

    point_new = transform(point)
    a_rel_new = transform(a_rel)
    a_centr_new = transform(a_centr)
    a_rot_new = transform(a_rot)
    a_trans_new = transform(a_trans)
    a_cor_new = transform(a_cor)
    a_abs_new = transform(a_abs)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=[point_new[0]], y=[point_new[1]], z=[point_new[2]],
                               mode='markers', marker=dict(color='red', size=8), name='Point M'))

    vectors = [a_rel_new, a_centr_new, a_rot_new, a_trans_new, a_cor_new, a_abs_new]
    colors = ['blue', 'green', 'orange', 'brown', 'cyan', 'purple']
    labels = ['a_rel', 'a_centr', 'a_rot', 'a_trans_post', 'a_cor', 'a_abs']
    numeric_vals = [np.linalg.norm(v) for v in [a_rel, a_centr, a_rot, a_trans, a_cor, a_abs]]

    for vec, col, lab, val in zip(vectors, colors, labels, numeric_vals):
        fig.add_trace(go.Scatter3d(x=[point_new[0], point_new[0]+vec[0]],
                                   y=[point_new[1], point_new[1]+vec[1]],
                                   z=[point_new[2], point_new[2]+vec[2]],
                                   mode='lines+text', line=dict(color=col, width=3),
                                   text=['', f'{lab} = {val:.2f}'], textposition='middle right', name=lab))
        end_x = point_new[0] + vec[0]
        end_y = point_new[1] + vec[1]
        end_z = point_new[2] + vec[2]
        arrow_scale = 0.15
        u = vec[0] * arrow_scale
        v = vec[1] * arrow_scale
        w = vec[2] * arrow_scale
        fig.add_trace(go.Cone(x=[end_x], y=[end_y], z=[end_z],
                              u=[u], v=[v], w=[w],
                              colorscale=[[0, col], [1, col]],
                              showscale=False, sizemode='scaled', sizeref=0.3,
                              showlegend=False))

    all_pts = [point_new] + [point_new+v for v in vectors] + [np.zeros(3)]
    all_arr = np.array(all_pts)
    min_vals = np.min(all_arr, axis=0)
    max_vals = np.max(all_arr, axis=0)
    max_range = np.max(max_vals - min_vals)
    axis_len = max_range * 0.25

    fig.add_trace(go.Scatter3d(x=[point_new[0], point_new[0]+axis_len], y=[point_new[1], point_new[1]],
                               z=[point_new[2], point_new[2]], mode='lines+text',
                               line=dict(color='black', width=2), text=['', 'a_отн⊥'], textposition='middle right', showlegend=False))
    fig.add_trace(go.Scatter3d(x=[point_new[0], point_new[0]], y=[point_new[1], point_new[1]+axis_len],
                               z=[point_new[2], point_new[2]], mode='lines+text',
                               line=dict(color='black', width=2), text=['', 'a_cor'], textposition='middle right', showlegend=False))
    fig.add_trace(go.Scatter3d(x=[point_new[0], point_new[0]], y=[point_new[1], point_new[1]],
                               z=[point_new[2], point_new[2]+axis_len], mode='lines+text',
                               line=dict(color='black', width=2), text=['', 'ω'], textposition='middle right', showlegend=False))

    fig.update_layout(title='Векторы ускорений в точке M (оси: ω, a_cor, a_отн⊥)',
                      scene=dict(xaxis_title="a_отн⊥", yaxis_title="a_cor", zaxis_title="ω", aspectmode='auto'),
                      legend=dict(x=0.8, y=0.9))
    return fig.to_json()
