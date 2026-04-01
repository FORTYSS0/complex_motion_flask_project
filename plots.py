import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch
import plotly.graph_objects as go

# ==================== Статичные PNG-графики для экспорта ====================

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def draw_axes(ax, origin=(0,0,0), length=None, color='black', labels=['X', 'Y', 'Z']):
    for i, label in enumerate(labels):
        vec = [0,0,0]
        vec[i] = length
        ax.quiver(origin[0], origin[1], origin[2],
                  vec[0], vec[1], vec[2],
                  color=color, label=label, arrow_length_ratio=0.03, linewidth=2)
        ax.text(origin[0]+vec[0], origin[1]+vec[1], origin[2]+vec[2], label,
                color=color, fontsize=10, ha='center', va='center')

def generate_all_plots(data):
    """Создаёт статичные PNG‑графики для экспорта."""
    # ---- 1. Траектория ----
    t_vals = np.linspace(0, 2.5, 200)
    x_t = 8 * np.cos(np.pi * t_vals**2 / 3)
    y_t = 16 * np.sin(np.pi * t_vals**2 / 3)
    z_t = 2 * t_vals**2 + 4

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_t, y_t, z_t, label='Траектория точки')
    ax.scatter(data['point'][0], data['point'][1], data['point'][2],
               color='red', s=50, label='Положение при t=1')
    ax.set_xlabel("X'")
    ax.set_ylabel("Y'")
    ax.set_zlabel("Z'")
    ax.set_title('Способ задания движения (абсолютная траектория)')

    O = np.array([0.0, 0.0, data['zp']])

    all_points = np.vstack([np.column_stack([x_t, y_t, z_t]),
                            [data['point']], [0,0,0], O])
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    max_range = np.max(max_vals - min_vals)
    axis_length = max_range * 0.25

    axis_ends = [[axis_length,0,0], [0,axis_length,0], [0,0,axis_length]]
    red_ends = [O + np.array([axis_length,0,0]),
                O + np.array([0,axis_length,0]),
                O + np.array([0,0,axis_length])]
    all_points = np.vstack([all_points, axis_ends, red_ends])
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    margin = 0.2
    ax.set_xlim([min_vals[0]-margin, max_vals[0]+margin])
    ax.set_ylim([min_vals[1]-margin, max_vals[1]+margin])
    ax.set_zlim([min_vals[2]-margin, max_vals[2]+margin])

    draw_axes(ax, origin=(0,0,0), length=axis_length, color='black',
              labels=["X'", "Y'", "Z'"])
    draw_axes(ax, origin=O, length=axis_length, color='red', labels=['X', 'Y', 'Z'])
    ax.scatter(O[0], O[1], O[2], color='blue', s=40, label='O (начало подвижной)')
    ax.text(O[0], O[1], O[2], ' O', color='blue', fontsize=10)

    V_abs = data['V_abs']
    V_norm = V_abs / (np.linalg.norm(V_abs) + 1e-8)
    arrow_length = axis_length * 0.4
    V_scaled = V_norm * arrow_length
    ax.quiver(data['point'][0], data['point'][1], data['point'][2],
              V_scaled[0], V_scaled[1], V_scaled[2],
              color='purple', label='V_abs', arrow_length_ratio=0.03)
    end = np.array(data['point']) + V_scaled
    offset = V_norm * 0.1
    text_pos = end + offset
    ax.text(text_pos[0], text_pos[1], text_pos[2],
            f'V_abs = {data["V_abs_mod"]:.2f}',
            color='purple', fontsize=8, ha='center', va='center')

    theta = np.linspace(0, -np.pi/2, 50)
    radius = axis_length * 0.6
    x_arc = O[0] + radius * np.cos(theta)
    y_arc = O[1] + radius * np.sin(theta)
    z_arc = O[2] + axis_length * 0.05
    z_arc_arr = np.full_like(x_arc, z_arc)
    ax.plot(x_arc, y_arc, z_arc_arr, color='green', linewidth=2)
    arrow = Arrow3D([x_arc[-2], x_arc[-1]], [y_arc[-2], y_arc[-1]],
                    [z_arc_arr[-2], z_arc_arr[-1]],
                    mutation_scale=20, lw=2, arrowstyle='->', color='green')
    ax.add_artist(arrow)
    mid = len(theta)//2
    ax.text(x_arc[mid], y_arc[mid], z_arc_arr[mid]+0.02, f'ω = {data["omega"]:.2f}',
            color='green', fontsize=9)

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'trajectory.png'), dpi=150)
    plt.close()

    # ---- 2. Скорости (статичные) ----
    point = data['point']
    vectors = [data['V_rel'], data['V_rot'], data['V_trans_post'], data['V_abs']]
    colors = ['blue', 'green', 'orange', 'purple']
    labels = ['V_rel', 'V_rot', 'V_trans_post', 'V_abs']
    numeric_vals = [data['V_rel_mod'], data['V_rot_mod'],
                    data['V_trans_post_mod'], data['V_abs_mod']]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Векторы скоростей в точке M')

    ax.scatter(point[0], point[1], point[2], color='red', s=50, label='Point M')
    for idx, (vec, color, label) in enumerate(zip(vectors, colors, labels)):
        ax.quiver(point[0], point[1], point[2],
                  vec[0], vec[1], vec[2],
                  color=color, label=label, arrow_length_ratio=0.03)
        if numeric_vals and idx < len(numeric_vals):
            end = np.array(point) + vec
            offset = vec / (np.linalg.norm(vec) + 1e-8) * 0.1
            text_pos = end + offset
            ax.text(text_pos[0], text_pos[1], text_pos[2],
                    f'{label} = {numeric_vals[idx]:.2f}',
                    color=color, fontsize=8, ha='center', va='center')

    all_points = [np.array(point)] + [np.array(point) + v for v in vectors] + [np.array([0,0,0])]
    all_arr = np.array(all_points)
    min_vals = np.min(all_arr, axis=0)
    max_vals = np.max(all_arr, axis=0)
    max_range = np.max(max_vals - min_vals)
    axis_length = max_range * 0.25
    axis_ends = [[axis_length,0,0], [0,axis_length,0], [0,0,axis_length]]
    all_arr = np.vstack([all_arr, axis_ends])
    min_vals = np.min(all_arr, axis=0)
    max_vals = np.max(all_arr, axis=0)
    margin = 0.2
    ax.set_xlim([min_vals[0]-margin, max_vals[0]+margin])
    ax.set_ylim([min_vals[1]-margin, max_vals[1]+margin])
    ax.set_zlim([min_vals[2]-margin, max_vals[2]+margin])

    draw_axes(ax, origin=(0,0,0), length=axis_length, color='black')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'velocities.png'), dpi=150)
    plt.close()

    # ---- 3. Ускорения (статичные) ----
    vectors = [data['a_rel'], data['a_centr'], data['a_rot'],
               data['a_trans_post'], data['a_cor'], data['a_abs']]
    colors = ['blue', 'green', 'orange', 'brown', 'cyan', 'purple']
    labels = ['a_rel', 'a_centr', 'a_rot', 'a_trans_post', 'a_cor', 'a_abs']
    numeric_vals = [data['a_rel_mod'], data['a_centr_mod'], data['a_rot_mod'],
                    data['a_trans_post_mod'], data['a_cor_mod'], data['a_abs_mod']]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Векторы ускорений в точке M')

    ax.scatter(point[0], point[1], point[2], color='red', s=50, label='Point M')
    for idx, (vec, color, label) in enumerate(zip(vectors, colors, labels)):
        ax.quiver(point[0], point[1], point[2],
                  vec[0], vec[1], vec[2],
                  color=color, label=label, arrow_length_ratio=0.03)
        if numeric_vals and idx < len(numeric_vals):
            end = np.array(point) + vec
            offset = vec / (np.linalg.norm(vec) + 1e-8) * 0.1
            text_pos = end + offset
            ax.text(text_pos[0], text_pos[1], text_pos[2],
                    f'{label} = {numeric_vals[idx]:.2f}',
                    color=color, fontsize=8, ha='center', va='center')

    all_points = [np.array(point)] + [np.array(point) + v for v in vectors] + [np.array([0,0,0])]
    all_arr = np.array(all_points)
    min_vals = np.min(all_arr, axis=0)
    max_vals = np.max(all_arr, axis=0)
    max_range = np.max(max_vals - min_vals)
    axis_length = max_range * 0.25
    axis_ends = [[axis_length,0,0], [0,axis_length,0], [0,0,axis_length]]
    all_arr = np.vstack([all_arr, axis_ends])
    min_vals = np.min(all_arr, axis=0)
    max_vals = np.max(all_arr, axis=0)
    margin = 0.2
    ax.set_xlim([min_vals[0]-margin, max_vals[0]+margin])
    ax.set_ylim([min_vals[1]-margin, max_vals[1]+margin])
    ax.set_zlim([min_vals[2]-margin, max_vals[2]+margin])

    draw_axes(ax, origin=(0,0,0), length=axis_length, color='black')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'accelerations.png'), dpi=150)
    plt.close()


# ==================== Интерактивные графики (Plotly) ====================

def generate_interactive_trajectory(data):
    """Возвращает JSON для интерактивного графика траектории с одной стрелкой направления в конце,
       увеличенной стрелкой ω (в начале дуги, развёрнутой в противоположную сторону) и горизонтальной проекцией траектории (жирная линия)."""
    t_vals = np.linspace(0, 2.5, 200)
    x_t = 8 * np.cos(np.pi * t_vals**2 / 3)
    y_t = 16 * np.sin(np.pi * t_vals**2 / 3)
    z_t = 2 * t_vals**2 + 4

    fig = go.Figure()

    # Траектория
    fig.add_trace(go.Scatter3d(
        x=x_t, y=y_t, z=z_t,
        mode='lines',
        line=dict(color='blue', width=4),
        name='Траектория'
    ))

    # Горизонтальная проекция траектории (жирная линия)
    z_const = data['point'][2]
    fig.add_trace(go.Scatter3d(
        x=x_t, y=y_t, z=np.full_like(z_t, z_const),
        mode='lines',
        line=dict(color='orange', width=4, dash='dash'),
        name='Проекция траектории (z = const)'
    ))

    # Стрелка направления в конце траектории (один конус)
    i_end = len(t_vals)-1
    dx = x_t[-1] - x_t[-2]
    dy = y_t[-1] - y_t[-2]
    dz = z_t[-1] - z_t[-2]
    length = np.sqrt(dx**2 + dy**2 + dz**2)
    if length > 1e-8:
        scale = 0.15 * max(np.ptp(x_t), np.ptp(y_t), np.ptp(z_t))
        u = dx / length * scale
        v = dy / length * scale
        w = dz / length * scale
        fig.add_trace(go.Cone(
            x=[x_t[-1]], y=[y_t[-1]], z=[z_t[-1]],
            u=[u], v=[v], w=[w],
            colorscale=[[0, 'green'], [1, 'green']],
            showscale=False,
            sizemode='scaled',
            sizeref=0.5,
            name='Направление движения'
        ))

    # Точка M
    fig.add_trace(go.Scatter3d(
        x=[data['point'][0]], y=[data['point'][1]], z=[data['point'][2]],
        mode='markers',
        marker=dict(color='red', size=8),
        name='M (t=1)'
    ))

    # Определяем длину осей
    all_points = np.vstack([np.column_stack([x_t, y_t, z_t]), [data['point']], [0,0,0]])
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    max_range = np.max(max_vals - min_vals)
    axis_len = max_range * 0.3

    # Неподвижные оси (чёрные) с конусами
    # Ось X'
    fig.add_trace(go.Scatter3d(x=[0, axis_len], y=[0,0], z=[0,0], mode='lines',
                               line=dict(color='black', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[axis_len], y=[0], z=[0], u=[axis_len*0.2], v=[0], w=[0],
                          colorscale=[[0,'black'],[1,'black']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[axis_len*0.9], y=[0], z=[0], mode='text', text=["X'"], textfont=dict(color='black', size=10), showlegend=False))
    # Ось Y'
    fig.add_trace(go.Scatter3d(x=[0,0], y=[0, axis_len], z=[0,0], mode='lines',
                               line=dict(color='black', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[0], y=[axis_len], z=[0], u=[0], v=[axis_len*0.2], w=[0],
                          colorscale=[[0,'black'],[1,'black']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0], y=[axis_len*0.9], z=[0], mode='text', text=["Y'"], textfont=dict(color='black', size=10), showlegend=False))
    # Ось Z'
    fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[0, axis_len], mode='lines',
                               line=dict(color='black', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[0], y=[0], z=[axis_len], u=[0], v=[0], w=[axis_len*0.2],
                          colorscale=[[0,'black'],[1,'black']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[axis_len*0.9], mode='text', text=["Z'"], textfont=dict(color='black', size=10), showlegend=False))

    # Подвижные оси (красные) из точки O
    O = np.array([0.0, 0.0, data['zp']])
    # Ось X
    fig.add_trace(go.Scatter3d(x=[O[0], O[0]+axis_len], y=[O[1], O[1]], z=[O[2], O[2]], mode='lines',
                               line=dict(color='red', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[O[0]+axis_len], y=[O[1]], z=[O[2]], u=[axis_len*0.2], v=[0], w=[0],
                          colorscale=[[0,'red'],[1,'red']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[O[0]+axis_len*0.9], y=[O[1]], z=[O[2]], mode='text', text=["X"], textfont=dict(color='red', size=10), showlegend=False))
    # Ось Y
    fig.add_trace(go.Scatter3d(x=[O[0], O[0]], y=[O[1], O[1]+axis_len], z=[O[2], O[2]], mode='lines',
                               line=dict(color='red', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[O[0]], y=[O[1]+axis_len], z=[O[2]], u=[0], v=[axis_len*0.2], w=[0],
                          colorscale=[[0,'red'],[1,'red']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[O[0]], y=[O[1]+axis_len*0.9], z=[O[2]], mode='text', text=["Y"], textfont=dict(color='red', size=10), showlegend=False))
    # Ось Z (ω)
    fig.add_trace(go.Scatter3d(x=[O[0], O[0]], y=[O[1], O[1]], z=[O[2], O[2]+axis_len], mode='lines',
                               line=dict(color='red', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[O[0]], y=[O[1]], z=[O[2]+axis_len], u=[0], v=[0], w=[axis_len*0.2],
                          colorscale=[[0,'red'],[1,'red']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[O[0]], y=[O[1]], z=[O[2]+axis_len*0.9], mode='text', text=["ω"], textfont=dict(color='red', size=10), showlegend=False))

    # Точка O
    fig.add_trace(go.Scatter3d(x=[O[0]], y=[O[1]], z=[O[2]], mode='markers',
                               marker=dict(color='blue', size=6), name='O (начало подвижной)'))

    # Вектор абсолютной скорости (масштабированный)
    V_abs = data['V_abs']
    V_norm = V_abs / (np.linalg.norm(V_abs) + 1e-8)
    arrow_len = axis_len * 0.4
    V_scaled = V_norm * arrow_len
    fig.add_trace(go.Scatter3d(x=[data['point'][0], data['point'][0]+V_scaled[0]],
                               y=[data['point'][1], data['point'][1]+V_scaled[1]],
                               z=[data['point'][2], data['point'][2]+V_scaled[2]],
                               mode='lines+text', line=dict(color='purple', width=3),
                               text=['', f'V_abs = {data["V_abs_mod"]:.2f}'], textposition='middle right', name='V_abs'))
    # Стрелка в конце V_abs
    end_x = data['point'][0] + V_scaled[0]
    end_y = data['point'][1] + V_scaled[1]
    end_z = data['point'][2] + V_scaled[2]
    fig.add_trace(go.Cone(
        x=[end_x], y=[end_y], z=[end_z],
        u=[V_scaled[0]*0.15], v=[V_scaled[1]*0.15], w=[V_scaled[2]*0.15],
        colorscale=[[0, 'purple'], [1, 'purple']],
        showscale=False,
        sizemode='scaled',
        sizeref=0.5,
        name='V_abs direction',
        showlegend=False
    ))

    # --- Дуга вращения (ω) с увеличенной стрелкой (конусом) в начале дуги, развёрнутой в противоположную сторону ---
    theta = np.linspace(0, -np.pi/2, 50)
    radius = axis_len * 0.6
    x_arc = O[0] + radius * np.cos(theta)
    y_arc = O[1] + radius * np.sin(theta)
    z_arc = O[2] + axis_len * 0.05
    fig.add_trace(go.Scatter3d(x=x_arc, y=y_arc, z=np.full_like(x_arc, z_arc), mode='lines',
                               line=dict(color='green', width=2), name='ω'))
    # Стрелка (конус) в начале дуги, развёрнутая
    start_x = x_arc[0]
    start_y = y_arc[0]
    start_z = z_arc
    dx_arc = x_arc[1] - x_arc[0]
    dy_arc = y_arc[1] - y_arc[0]
    dz_arc = 0
    len_arc = np.sqrt(dx_arc**2 + dy_arc**2 + dz_arc**2)
    if len_arc > 1e-8:
        arrow_scale_arc = 0.3 * radius
        u_arc = -dx_arc / len_arc * arrow_scale_arc
        v_arc = -dy_arc / len_arc * arrow_scale_arc
        w_arc = 0
        fig.add_trace(go.Cone(
            x=[start_x], y=[start_y], z=[start_z],
            u=[u_arc], v=[v_arc], w=[w_arc],
            colorscale=[[0, 'green'], [1, 'green']],
            showscale=False,
            sizemode='scaled',
            sizeref=0.3,
            name='ω direction'
        ))
    # Подпись ω (без знака минус)
    mid = len(theta)//2
    fig.add_trace(go.Scatter3d(x=[x_arc[mid]], y=[y_arc[mid]], z=[z_arc + 0.02], mode='text',
                               text=[f'ω = {abs(data["omega"]):.2f}'], textfont=dict(color='green', size=10), showlegend=False))

    fig.update_layout(title='Способ задания движения (абсолютная траектория)',
                      scene=dict(xaxis_title="X'", yaxis_title="Y'", zaxis_title="Z'", aspectmode='auto'),
                      legend=dict(x=0.8, y=0.9))
    return fig.to_json()


def generate_interactive_velocities(data):
    """Интерактивный график скоростей с конусами на концах осей и векторов."""
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

    def transform(v):
        return R @ v

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
        # Стрелка-конус в конце
        end_x = point_new[0] + vec[0]
        end_y = point_new[1] + vec[1]
        end_z = point_new[2] + vec[2]
        arrow_scale = 0.15
        u = vec[0] * arrow_scale
        v = vec[1] * arrow_scale
        w = vec[2] * arrow_scale
        fig.add_trace(go.Cone(
            x=[end_x], y=[end_y], z=[end_z],
            u=[u], v=[v], w=[w],
            colorscale=[[0, col], [1, col]],
            showscale=False,
            sizemode='scaled',
            sizeref=0.3,
            name=f'{lab} direction',
            showlegend=False
        ))

    # Оси новой системы (чёрные) с конусами
    all_pts = [point_new] + [point_new+v for v in vectors] + [np.zeros(3)]
    all_arr = np.array(all_pts)
    min_vals = np.min(all_arr, axis=0)
    max_vals = np.max(all_arr, axis=0)
    max_range = np.max(max_vals - min_vals)
    axis_len = max_range * 0.25

    # Ось X (V_отн)
    fig.add_trace(go.Scatter3d(x=[point_new[0], point_new[0]+axis_len], y=[point_new[1], point_new[1]],
                               z=[point_new[2], point_new[2]], mode='lines',
                               line=dict(color='black', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[point_new[0]+axis_len], y=[point_new[1]], z=[point_new[2]],
                          u=[axis_len*0.2], v=[0], w=[0],
                          colorscale=[[0,'black'],[1,'black']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[point_new[0]+axis_len*0.9], y=[point_new[1]], z=[point_new[2]],
                               mode='text', text=['V_отн'], textfont=dict(color='black', size=10), showlegend=False))
    # Ось Y (a_cor)
    fig.add_trace(go.Scatter3d(x=[point_new[0], point_new[0]], y=[point_new[1], point_new[1]+axis_len],
                               z=[point_new[2], point_new[2]], mode='lines',
                               line=dict(color='black', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[point_new[0]], y=[point_new[1]+axis_len], z=[point_new[2]],
                          u=[0], v=[axis_len*0.2], w=[0],
                          colorscale=[[0,'black'],[1,'black']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[point_new[0]], y=[point_new[1]+axis_len*0.9], z=[point_new[2]],
                               mode='text', text=['a_cor'], textfont=dict(color='black', size=10), showlegend=False))
    # Ось Z (ω)
    fig.add_trace(go.Scatter3d(x=[point_new[0], point_new[0]], y=[point_new[1], point_new[1]],
                               z=[point_new[2], point_new[2]+axis_len], mode='lines',
                               line=dict(color='black', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[point_new[0]], y=[point_new[1]], z=[point_new[2]+axis_len],
                          u=[0], v=[0], w=[axis_len*0.2],
                          colorscale=[[0,'black'],[1,'black']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[point_new[0]], y=[point_new[1]], z=[point_new[2]+axis_len*0.9],
                               mode='text', text=['ω'], textfont=dict(color='black', size=10), showlegend=False))

    fig.update_layout(title='Векторы скоростей в точке M (оси: V_отн, a_cor, ω)',
                      scene=dict(xaxis_title="V_отн", yaxis_title="a_cor", zaxis_title="ω", aspectmode='auto'),
                      legend=dict(x=0.8, y=0.9))
    return fig.to_json()


def generate_interactive_accelerations(data):
    """Интерактивный график ускорений с конусами на концах осей и векторов."""
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

    def transform(v):
        return R @ v

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
        # Линия вектора
        fig.add_trace(go.Scatter3d(x=[point_new[0], point_new[0]+vec[0]],
                                   y=[point_new[1], point_new[1]+vec[1]],
                                   z=[point_new[2], point_new[2]+vec[2]],
                                   mode='lines+text', line=dict(color=col, width=3),
                                   text=['', f'{lab} = {val:.2f}'], textposition='middle right', name=lab))
        # Стрелка-конус в конце
        end_x = point_new[0] + vec[0]
        end_y = point_new[1] + vec[1]
        end_z = point_new[2] + vec[2]
        arrow_scale = 0.15
        u = vec[0] * arrow_scale
        v = vec[1] * arrow_scale
        w = vec[2] * arrow_scale
        fig.add_trace(go.Cone(
            x=[end_x], y=[end_y], z=[end_z],
            u=[u], v=[v], w=[w],
            colorscale=[[0, col], [1, col]],
            showscale=False,
            sizemode='scaled',
            sizeref=0.3,
            name=f'{lab} direction',
            showlegend=False
        ))

    # Оси новой системы (чёрные) с конусами
    all_pts = [point_new] + [point_new+v for v in vectors] + [np.zeros(3)]
    all_arr = np.array(all_pts)
    min_vals = np.min(all_arr, axis=0)
    max_vals = np.max(all_arr, axis=0)
    max_range = np.max(max_vals - min_vals)
    axis_len = max_range * 0.25

    # Ось X (a_отн⊥)
    fig.add_trace(go.Scatter3d(x=[point_new[0], point_new[0]+axis_len], y=[point_new[1], point_new[1]],
                               z=[point_new[2], point_new[2]], mode='lines',
                               line=dict(color='black', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[point_new[0]+axis_len], y=[point_new[1]], z=[point_new[2]],
                          u=[axis_len*0.2], v=[0], w=[0],
                          colorscale=[[0,'black'],[1,'black']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[point_new[0]+axis_len*0.9], y=[point_new[1]], z=[point_new[2]],
                               mode='text', text=['a_отн⊥'], textfont=dict(color='black', size=10), showlegend=False))
    # Ось Y (a_cor)
    fig.add_trace(go.Scatter3d(x=[point_new[0], point_new[0]], y=[point_new[1], point_new[1]+axis_len],
                               z=[point_new[2], point_new[2]], mode='lines',
                               line=dict(color='black', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[point_new[0]], y=[point_new[1]+axis_len], z=[point_new[2]],
                          u=[0], v=[axis_len*0.2], w=[0],
                          colorscale=[[0,'black'],[1,'black']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[point_new[0]], y=[point_new[1]+axis_len*0.9], z=[point_new[2]],
                               mode='text', text=['a_cor'], textfont=dict(color='black', size=10), showlegend=False))
    # Ось Z (ω)
    fig.add_trace(go.Scatter3d(x=[point_new[0], point_new[0]], y=[point_new[1], point_new[1]],
                               z=[point_new[2], point_new[2]+axis_len], mode='lines',
                               line=dict(color='black', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[point_new[0]], y=[point_new[1]], z=[point_new[2]+axis_len],
                          u=[0], v=[0], w=[axis_len*0.2],
                          colorscale=[[0,'black'],[1,'black']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[point_new[0]], y=[point_new[1]], z=[point_new[2]+axis_len*0.9],
                               mode='text', text=['ω'], textfont=dict(color='black', size=10), showlegend=False))

    fig.update_layout(title='Векторы ускорений в точке M (оси: ω, a_cor, a_отн⊥)',
                      scene=dict(xaxis_title="a_отн⊥", yaxis_title="a_cor", zaxis_title="ω", aspectmode='auto'),
                      legend=dict(x=0.8, y=0.9))
    return fig.to_json()


# ==================== Новые комбинированные графики ====================

def generate_interactive_trajectory_with_velocities(data):
    """
    Рисунок 1 (траектория) + векторы скоростей (V_rel, V_rot, V_trans_post, V_abs)
    без фиолетового вектора V_abs с рисунка 1. Все векторы масштабированы до одинаковой длины,
    указывающей только направление.
    """
    t_vals = np.linspace(0, 2.5, 200)
    x_t = 8 * np.cos(np.pi * t_vals**2 / 3)
    y_t = 16 * np.sin(np.pi * t_vals**2 / 3)
    z_t = 2 * t_vals**2 + 4

    fig = go.Figure()

    # Траектория
    fig.add_trace(go.Scatter3d(
        x=x_t, y=y_t, z=z_t,
        mode='lines',
        line=dict(color='blue', width=4),
        name='Траектория'
    ))

    # Горизонтальная проекция траектории
    z_const = data['point'][2]
    fig.add_trace(go.Scatter3d(
        x=x_t, y=y_t, z=np.full_like(z_t, z_const),
        mode='lines',
        line=dict(color='orange', width=4, dash='dash'),
        name='Проекция траектории (z = const)'
    ))

    # Точка M
    fig.add_trace(go.Scatter3d(
        x=[data['point'][0]], y=[data['point'][1]], z=[data['point'][2]],
        mode='markers',
        marker=dict(color='red', size=8),
        name='M (t=1)'
    ))

    # Определяем длину осей (как на первом графике)
    all_points = np.vstack([np.column_stack([x_t, y_t, z_t]), [data['point']], [0,0,0]])
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    max_range = np.max(max_vals - min_vals)
    axis_len = max_range * 0.3

    # Неподвижные оси (чёрные) с конусами (как на первом графике)
    # Ось X'
    fig.add_trace(go.Scatter3d(x=[0, axis_len], y=[0,0], z=[0,0], mode='lines',
                               line=dict(color='black', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[axis_len], y=[0], z=[0], u=[axis_len*0.2], v=[0], w=[0],
                          colorscale=[[0,'black'],[1,'black']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[axis_len*0.9], y=[0], z=[0], mode='text', text=["X'"], textfont=dict(color='black', size=10), showlegend=False))
    # Ось Y'
    fig.add_trace(go.Scatter3d(x=[0,0], y=[0, axis_len], z=[0,0], mode='lines',
                               line=dict(color='black', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[0], y=[axis_len], z=[0], u=[0], v=[axis_len*0.2], w=[0],
                          colorscale=[[0,'black'],[1,'black']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0], y=[axis_len*0.9], z=[0], mode='text', text=["Y'"], textfont=dict(color='black', size=10), showlegend=False))
    # Ось Z'
    fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[0, axis_len], mode='lines',
                               line=dict(color='black', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[0], y=[0], z=[axis_len], u=[0], v=[0], w=[axis_len*0.2],
                          colorscale=[[0,'black'],[1,'black']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[axis_len*0.9], mode='text', text=["Z'"], textfont=dict(color='black', size=10), showlegend=False))

    # Подвижные оси (красные) из точки O
    O = np.array([0.0, 0.0, data['zp']])
    # Ось X
    fig.add_trace(go.Scatter3d(x=[O[0], O[0]+axis_len], y=[O[1], O[1]], z=[O[2], O[2]], mode='lines',
                               line=dict(color='red', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[O[0]+axis_len], y=[O[1]], z=[O[2]], u=[axis_len*0.2], v=[0], w=[0],
                          colorscale=[[0,'red'],[1,'red']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[O[0]+axis_len*0.9], y=[O[1]], z=[O[2]], mode='text', text=["X"], textfont=dict(color='red', size=10), showlegend=False))
    # Ось Y
    fig.add_trace(go.Scatter3d(x=[O[0], O[0]], y=[O[1], O[1]+axis_len], z=[O[2], O[2]], mode='lines',
                               line=dict(color='red', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[O[0]], y=[O[1]+axis_len], z=[O[2]], u=[0], v=[axis_len*0.2], w=[0],
                          colorscale=[[0,'red'],[1,'red']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[O[0]], y=[O[1]+axis_len*0.9], z=[O[2]], mode='text', text=["Y"], textfont=dict(color='red', size=10), showlegend=False))
    # Ось Z (ω)
    fig.add_trace(go.Scatter3d(x=[O[0], O[0]], y=[O[1], O[1]], z=[O[2], O[2]+axis_len], mode='lines',
                               line=dict(color='red', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[O[0]], y=[O[1]], z=[O[2]+axis_len], u=[0], v=[0], w=[axis_len*0.2],
                          colorscale=[[0,'red'],[1,'red']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[O[0]], y=[O[1]], z=[O[2]+axis_len*0.9], mode='text', text=["ω"], textfont=dict(color='red', size=10), showlegend=False))

    # Точка O
    fig.add_trace(go.Scatter3d(x=[O[0]], y=[O[1]], z=[O[2]], mode='markers',
                               marker=dict(color='blue', size=6), name='O (начало подвижной)'))

    # --- Дуга вращения (ω) (как на первом графике) ---
    theta = np.linspace(0, -np.pi/2, 50)
    radius = axis_len * 0.6
    x_arc = O[0] + radius * np.cos(theta)
    y_arc = O[1] + radius * np.sin(theta)
    z_arc = O[2] + axis_len * 0.05
    fig.add_trace(go.Scatter3d(x=x_arc, y=y_arc, z=np.full_like(x_arc, z_arc), mode='lines',
                               line=dict(color='green', width=2), name='ω'))
    # Стрелка в начале дуги, развёрнутая
    start_x = x_arc[0]
    start_y = y_arc[0]
    start_z = z_arc
    dx_arc = x_arc[1] - x_arc[0]
    dy_arc = y_arc[1] - y_arc[0]
    len_arc = np.sqrt(dx_arc**2 + dy_arc**2)
    if len_arc > 1e-8:
        arrow_scale_arc = 0.3 * radius
        u_arc = -dx_arc / len_arc * arrow_scale_arc
        v_arc = -dy_arc / len_arc * arrow_scale_arc
        fig.add_trace(go.Cone(x=[start_x], y=[start_y], z=[start_z], u=[u_arc], v=[v_arc], w=[0],
                              colorscale=[[0,'green'],[1,'green']], showscale=False, sizemode='scaled', sizeref=0.3, name='ω direction'))
    # Подпись ω
    mid = len(theta)//2
    fig.add_trace(go.Scatter3d(x=[x_arc[mid]], y=[y_arc[mid]], z=[z_arc + 0.02], mode='text',
                               text=[f'ω = {abs(data["omega"]):.2f}'], textfont=dict(color='green', size=10), showlegend=False))

    # --- Векторы скоростей (из data) – все масштабированы до одинаковой длины ---
    vectors = [data['V_rel'], data['V_rot'], data['V_trans_post'], data['V_abs']]
    colors = ['blue', 'green', 'orange', 'purple']
    labels = ['V_rel', 'V_rot', 'V_trans_post', 'V_abs']
    # Вычисляем масштаб для всех векторов – они будут иметь длину 0.3 * axis_len
    scale_len = axis_len * 0.3

    for vec, col, lab in zip(vectors, colors, labels):
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            unit = vec / norm
            scaled_vec = unit * scale_len
        else:
            scaled_vec = np.zeros(3)
        # Линия вектора
        fig.add_trace(go.Scatter3d(x=[data['point'][0], data['point'][0]+scaled_vec[0]],
                                   y=[data['point'][1], data['point'][1]+scaled_vec[1]],
                                   z=[data['point'][2], data['point'][2]+scaled_vec[2]],
                                   mode='lines+text', line=dict(color=col, width=3),
                                   text=['', f'{lab}'], textposition='middle right', name=lab))
        # Стрелка-конус в конце
        end_x = data['point'][0] + scaled_vec[0]
        end_y = data['point'][1] + scaled_vec[1]
        end_z = data['point'][2] + scaled_vec[2]
        arrow_scale = 0.15
        u = scaled_vec[0] * arrow_scale
        v = scaled_vec[1] * arrow_scale
        w = scaled_vec[2] * arrow_scale
        fig.add_trace(go.Cone(x=[end_x], y=[end_y], z=[end_z], u=[u], v=[v], w=[w],
                              colorscale=[[0,col],[1,col]], showscale=False, sizemode='scaled', sizeref=0.3,
                              name=f'{lab} direction', showlegend=False))

    # --- Настройка сцены ---
    # Для авто-масштабирования просто оставляем scene без явных лимитов
    fig.update_layout(title='Траектория и векторы скоростей (исходная система, векторы масштабированы)',
                      scene=dict(xaxis_title="X'", yaxis_title="Y'", zaxis_title="Z'", aspectmode='auto'),
                      legend=dict(x=0.8, y=0.9))
    return fig.to_json()


def generate_interactive_trajectory_with_accelerations(data):
    """
    Рисунок 1 (траектория) + векторы ускорений (a_rel, a_centr, a_rot, a_trans_post, a_cor, a_abs)
    без фиолетового вектора V_abs с рисунка 1. Все векторы масштабированы до одинаковой длины,
    указывающей только направление.
    """
    t_vals = np.linspace(0, 2.5, 200)
    x_t = 8 * np.cos(np.pi * t_vals**2 / 3)
    y_t = 16 * np.sin(np.pi * t_vals**2 / 3)
    z_t = 2 * t_vals**2 + 4

    fig = go.Figure()

    # Траектория
    fig.add_trace(go.Scatter3d(
        x=x_t, y=y_t, z=z_t,
        mode='lines',
        line=dict(color='blue', width=4),
        name='Траектория'
    ))

    # Горизонтальная проекция траектории
    z_const = data['point'][2]
    fig.add_trace(go.Scatter3d(
        x=x_t, y=y_t, z=np.full_like(z_t, z_const),
        mode='lines',
        line=dict(color='orange', width=4, dash='dash'),
        name='Проекция траектории (z = const)'
    ))

    # Точка M
    fig.add_trace(go.Scatter3d(
        x=[data['point'][0]], y=[data['point'][1]], z=[data['point'][2]],
        mode='markers',
        marker=dict(color='red', size=8),
        name='M (t=1)'
    ))

    # Определяем длину осей (как на первом графике)
    all_points = np.vstack([np.column_stack([x_t, y_t, z_t]), [data['point']], [0,0,0]])
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    max_range = np.max(max_vals - min_vals)
    axis_len = max_range * 0.3

    # Неподвижные оси (чёрные) с конусами (как на первом графике)
    # Ось X'
    fig.add_trace(go.Scatter3d(x=[0, axis_len], y=[0,0], z=[0,0], mode='lines',
                               line=dict(color='black', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[axis_len], y=[0], z=[0], u=[axis_len*0.2], v=[0], w=[0],
                          colorscale=[[0,'black'],[1,'black']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[axis_len*0.9], y=[0], z=[0], mode='text', text=["X'"], textfont=dict(color='black', size=10), showlegend=False))
    # Ось Y'
    fig.add_trace(go.Scatter3d(x=[0,0], y=[0, axis_len], z=[0,0], mode='lines',
                               line=dict(color='black', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[0], y=[axis_len], z=[0], u=[0], v=[axis_len*0.2], w=[0],
                          colorscale=[[0,'black'],[1,'black']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0], y=[axis_len*0.9], z=[0], mode='text', text=["Y'"], textfont=dict(color='black', size=10), showlegend=False))
    # Ось Z'
    fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[0, axis_len], mode='lines',
                               line=dict(color='black', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[0], y=[0], z=[axis_len], u=[0], v=[0], w=[axis_len*0.2],
                          colorscale=[[0,'black'],[1,'black']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[axis_len*0.9], mode='text', text=["Z'"], textfont=dict(color='black', size=10), showlegend=False))

    # Подвижные оси (красные) из точки O
    O = np.array([0.0, 0.0, data['zp']])
    # Ось X
    fig.add_trace(go.Scatter3d(x=[O[0], O[0]+axis_len], y=[O[1], O[1]], z=[O[2], O[2]], mode='lines',
                               line=dict(color='red', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[O[0]+axis_len], y=[O[1]], z=[O[2]], u=[axis_len*0.2], v=[0], w=[0],
                          colorscale=[[0,'red'],[1,'red']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[O[0]+axis_len*0.9], y=[O[1]], z=[O[2]], mode='text', text=["X"], textfont=dict(color='red', size=10), showlegend=False))
    # Ось Y
    fig.add_trace(go.Scatter3d(x=[O[0], O[0]], y=[O[1], O[1]+axis_len], z=[O[2], O[2]], mode='lines',
                               line=dict(color='red', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[O[0]], y=[O[1]+axis_len], z=[O[2]], u=[0], v=[axis_len*0.2], w=[0],
                          colorscale=[[0,'red'],[1,'red']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[O[0]], y=[O[1]+axis_len*0.9], z=[O[2]], mode='text', text=["Y"], textfont=dict(color='red', size=10), showlegend=False))
    # Ось Z (ω)
    fig.add_trace(go.Scatter3d(x=[O[0], O[0]], y=[O[1], O[1]], z=[O[2], O[2]+axis_len], mode='lines',
                               line=dict(color='red', width=2), showlegend=False))
    fig.add_trace(go.Cone(x=[O[0]], y=[O[1]], z=[O[2]+axis_len], u=[0], v=[0], w=[axis_len*0.2],
                          colorscale=[[0,'red'],[1,'red']], showscale=False, sizemode='scaled', sizeref=0.3, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[O[0]], y=[O[1]], z=[O[2]+axis_len*0.9], mode='text', text=["ω"], textfont=dict(color='red', size=10), showlegend=False))

    # Точка O
    fig.add_trace(go.Scatter3d(x=[O[0]], y=[O[1]], z=[O[2]], mode='markers',
                               marker=dict(color='blue', size=6), name='O (начало подвижной)'))

    # --- Дуга вращения (ω) (как на первом графике) ---
    theta = np.linspace(0, -np.pi/2, 50)
    radius = axis_len * 0.6
    x_arc = O[0] + radius * np.cos(theta)
    y_arc = O[1] + radius * np.sin(theta)
    z_arc = O[2] + axis_len * 0.05
    fig.add_trace(go.Scatter3d(x=x_arc, y=y_arc, z=np.full_like(x_arc, z_arc), mode='lines',
                               line=dict(color='green', width=2), name='ω'))
    # Стрелка в начале дуги, развёрнутая
    start_x = x_arc[0]
    start_y = y_arc[0]
    start_z = z_arc
    dx_arc = x_arc[1] - x_arc[0]
    dy_arc = y_arc[1] - y_arc[0]
    len_arc = np.sqrt(dx_arc**2 + dy_arc**2)
    if len_arc > 1e-8:
        arrow_scale_arc = 0.3 * radius
        u_arc = -dx_arc / len_arc * arrow_scale_arc
        v_arc = -dy_arc / len_arc * arrow_scale_arc
        fig.add_trace(go.Cone(x=[start_x], y=[start_y], z=[start_z], u=[u_arc], v=[v_arc], w=[0],
                              colorscale=[[0,'green'],[1,'green']], showscale=False, sizemode='scaled', sizeref=0.3, name='ω direction'))
    # Подпись ω
    mid = len(theta)//2
    fig.add_trace(go.Scatter3d(x=[x_arc[mid]], y=[y_arc[mid]], z=[z_arc + 0.02], mode='text',
                               text=[f'ω = {abs(data["omega"]):.2f}'], textfont=dict(color='green', size=10), showlegend=False))

    # --- Векторы ускорений (из data) – все масштабированы до одинаковой длины ---
    vectors = [data['a_rel'], data['a_centr'], data['a_rot'],
               data['a_trans_post'], data['a_cor'], data['a_abs']]
    colors = ['blue', 'green', 'orange', 'brown', 'cyan', 'purple']
    labels = ['a_rel', 'a_centr', 'a_rot', 'a_trans_post', 'a_cor', 'a_abs']
    scale_len = axis_len * 0.3

    for vec, col, lab in zip(vectors, colors, labels):
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            unit = vec / norm
            scaled_vec = unit * scale_len
        else:
            scaled_vec = np.zeros(3)
        # Линия вектора
        fig.add_trace(go.Scatter3d(x=[data['point'][0], data['point'][0]+scaled_vec[0]],
                                   y=[data['point'][1], data['point'][1]+scaled_vec[1]],
                                   z=[data['point'][2], data['point'][2]+scaled_vec[2]],
                                   mode='lines+text', line=dict(color=col, width=3),
                                   text=['', f'{lab}'], textposition='middle right', name=lab))
        # Стрелка-конус в конце
        end_x = data['point'][0] + scaled_vec[0]
        end_y = data['point'][1] + scaled_vec[1]
        end_z = data['point'][2] + scaled_vec[2]
        arrow_scale = 0.15
        u = scaled_vec[0] * arrow_scale
        v = scaled_vec[1] * arrow_scale
        w = scaled_vec[2] * arrow_scale
        fig.add_trace(go.Cone(x=[end_x], y=[end_y], z=[end_z], u=[u], v=[v], w=[w],
                              colorscale=[[0,col],[1,col]], showscale=False, sizemode='scaled', sizeref=0.3,
                              name=f'{lab} direction', showlegend=False))

    # --- Настройка сцены ---
    fig.update_layout(title='Траектория и векторы ускорений (исходная система, векторы масштабированы)',
                      scene=dict(xaxis_title="X'", yaxis_title="Y'", zaxis_title="Z'", aspectmode='auto'),
                      legend=dict(x=0.8, y=0.9))
    return fig.to_json()
