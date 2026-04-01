import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def generate_interactive_trajectory(data):
    """Возвращает JSON для интерактивного графика траектории."""
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

    # Точка M
    fig.add_trace(go.Scatter3d(
        x=[data['point'][0]], y=[data['point'][1]], z=[data['point'][2]],
        mode='markers',
        marker=dict(color='red', size=8),
        name='M (t=1)'
    ))

    # Неподвижные оси (чёрные)
    axis_len = 3.0  # подберите визуально, можно вычислить как раньше
    # Ось X'
    fig.add_trace(go.Scatter3d(
        x=[0, axis_len], y=[0,0], z=[0,0],
        mode='lines+text',
        line=dict(color='black', width=2),
        text=['', "X'"], textposition='middle right',
        showlegend=False
    ))
    # Ось Y'
    fig.add_trace(go.Scatter3d(
        x=[0,0], y=[0, axis_len], z=[0,0],
        mode='lines+text',
        line=dict(color='black', width=2),
        text=['', "Y'"], textposition='middle right',
        showlegend=False
    ))
    # Ось Z'
    fig.add_trace(go.Scatter3d(
        x=[0,0], y=[0,0], z=[0, axis_len],
        mode='lines+text',
        line=dict(color='black', width=2),
        text=['', "Z'"], textposition='middle right',
        showlegend=False
    ))

    # Подвижные оси (красные) из точки O
    O = np.array([0.0, 0.0, data['zp']])
    # Ось X
    fig.add_trace(go.Scatter3d(
        x=[O[0], O[0]+axis_len], y=[O[1], O[1]], z=[O[2], O[2]],
        mode='lines+text',
        line=dict(color='red', width=2),
        text=['', "X"], textposition='middle right',
        showlegend=False
    ))
    # Ось Y
    fig.add_trace(go.Scatter3d(
        x=[O[0], O[0]], y=[O[1], O[1]+axis_len], z=[O[2], O[2]],
        mode='lines+text',
        line=dict(color='red', width=2),
        text=['', "Y"], textposition='middle right',
        showlegend=False
    ))
    # Ось Z
    fig.add_trace(go.Scatter3d(
        x=[O[0], O[0]], y=[O[1], O[1]], z=[O[2], O[2]+axis_len],
        mode='lines+text',
        line=dict(color='red', width=2),
        text=['', "Z"], textposition='middle right',
        showlegend=False
    ))

    # Точка O
    fig.add_trace(go.Scatter3d(
        x=[O[0]], y=[O[1]], z=[O[2]],
        mode='markers',
        marker=dict(color='blue', size=6),
        name='O (начало подвижной)'
    ))

    # Вектор абсолютной скорости (масштабированный)
    V_abs = data['V_abs']
    V_norm = V_abs / np.linalg.norm(V_abs)
    arrow_len = axis_len * 0.4
    V_scaled = V_norm * arrow_len
    fig.add_trace(go.Scatter3d(
        x=[data['point'][0], data['point'][0]+V_scaled[0]],
        y=[data['point'][1], data['point'][1]+V_scaled[1]],
        z=[data['point'][2], data['point'][2]+V_scaled[2]],
        mode='lines+text',
        line=dict(color='purple', width=3),
        text=['', f'V_abs = {data["V_abs_mod"]:.2f}'],
        textposition='middle right',
        name='V_abs'
    ))

    # Дуга вращения (зелёная)
    theta = np.linspace(0, -np.pi/2, 50)
    radius = axis_len * 0.6
    x_arc = O[0] + radius * np.cos(theta)
    y_arc = O[1] + radius * np.sin(theta)
    z_arc = O[2] + axis_len * 0.05
    fig.add_trace(go.Scatter3d(
        x=x_arc, y=y_arc, z=z_arc,
        mode='lines',
        line=dict(color='green', width=2),
        name='ω'
    ))
    # Стрелка в конце дуги (добавляем короткую линию)
    fig.add_trace(go.Scatter3d(
        x=[x_arc[-2], x_arc[-1]],
        y=[y_arc[-2], y_arc[-1]],
        z=[z_arc[-2], z_arc[-1]],
        mode='lines',
        line=dict(color='green', width=3),
        showlegend=False
    ))
    # Текст ω
    mid = len(theta)//2
    fig.add_annotation(
        x=x_arc[mid], y=y_arc[mid], z=z_arc[mid]+0.02,
        text=f'ω = {data["omega"]:.2f}',
        showarrow=False,
        font=dict(color='green', size=10)
    )

    fig.update_layout(
        title='Способ задания движения (абсолютная траектория)',
        scene=dict(
            xaxis_title="X'",
            yaxis_title="Y'",
            zaxis_title="Z'",
            aspectmode='auto'
        ),
        legend=dict(x=0.8, y=0.9)
    )
    return fig.to_json()

def generate_interactive_velocities(data):
    """Интерактивный график скоростей в новой системе координат (V_отн, a_cor, ω)."""
    point = data['point']
    # Векторы в исходной системе
    V_rel = data['V_rel']
    V_rot = data['V_rot']
    V_trans = data['V_trans_post']
    V_abs = data['V_abs']

    # Новые ортонормированные базисные векторы
    e_x = V_rel / np.linalg.norm(V_rel)          # V_отн
    e_z = np.array([0.0, 0.0, 1.0])              # ω
    e_y = np.cross(e_z, e_x)                     # a_cor
    e_y = e_y / np.linalg.norm(e_y)
    R = np.array([e_x, e_y, e_z])

    def transform(v):
        return R @ v

    point_new = transform(point)
    V_rel_new = transform(V_rel)
    V_rot_new = transform(V_rot)
    V_trans_new = transform(V_trans)
    V_abs_new = transform(V_abs)

    fig = go.Figure()

    # Точка M
    fig.add_trace(go.Scatter3d(
        x=[point_new[0]], y=[point_new[1]], z=[point_new[2]],
        mode='markers',
        marker=dict(color='red', size=8),
        name='Point M'
    ))

    # Векторы скоростей
    vectors = [V_rel_new, V_rot_new, V_trans_new, V_abs_new]
    colors = ['blue', 'green', 'orange', 'purple']
    labels = ['V_rel', 'V_rot', 'V_trans_post', 'V_abs']
    numeric_vals = [np.linalg.norm(V_rel), np.linalg.norm(V_rot),
                    np.linalg.norm(V_trans), np.linalg.norm(V_abs)]
    for vec, col, lab, val in zip(vectors, colors, labels, numeric_vals):
        fig.add_trace(go.Scatter3d(
            x=[point_new[0], point_new[0]+vec[0]],
            y=[point_new[1], point_new[1]+vec[1]],
            z=[point_new[2], point_new[2]+vec[2]],
            mode='lines+text',
            line=dict(color=col, width=3),
            text=['', f'{lab} = {val:.2f}'],
            textposition='middle right',
            name=lab
        ))

    # Оси новой системы (чёрные)
    axis_len = 2.0  # подберите визуально
    # Ось X (V_отн)
    fig.add_trace(go.Scatter3d(
        x=[point_new[0], point_new[0]+axis_len],
        y=[point_new[1], point_new[1]],
        z=[point_new[2], point_new[2]],
        mode='lines+text',
        line=dict(color='black', width=2),
        text=['', 'V_отн'],
        textposition='middle right',
        showlegend=False
    ))
    # Ось Y (a_cor)
    fig.add_trace(go.Scatter3d(
        x=[point_new[0], point_new[0]],
        y=[point_new[1], point_new[1]+axis_len],
        z=[point_new[2], point_new[2]],
        mode='lines+text',
        line=dict(color='black', width=2),
        text=['', 'a_cor'],
        textposition='middle right',
        showlegend=False
    ))
    # Ось Z (ω)
    fig.add_trace(go.Scatter3d(
        x=[point_new[0], point_new[0]],
        y=[point_new[1], point_new[1]],
        z=[point_new[2], point_new[2]+axis_len],
        mode='lines+text',
        line=dict(color='black', width=2),
        text=['', 'ω'],
        textposition='middle right',
        showlegend=False
    ))

    fig.update_layout(
        title='Векторы скоростей в точке M (оси: V_отн, a_cor, ω)',
        scene=dict(
            xaxis_title="V_отн",
            yaxis_title="a_cor",
            zaxis_title="ω",
            aspectmode='auto'
        ),
        legend=dict(x=0.8, y=0.9)
    )
    return fig.to_json()

def generate_interactive_accelerations(data):
    """Интерактивный график ускорений в новой системе координат (ω, a_cor, a_отн)."""
    point = data['point']
    # Векторы
    a_rel = data['a_rel']
    a_centr = data['a_centr']
    a_rot = data['a_rot']
    a_trans = data['a_trans_post']
    a_cor = data['a_cor']
    a_abs = data['a_abs']

    # Базис: e_z = ω, e_y = a_cor, e_x = e_y × e_z
    e_z = np.array([0.0, 0.0, 1.0])
    e_y = a_cor / np.linalg.norm(a_cor)
    e_x = np.cross(e_y, e_z)
    e_x = e_x / np.linalg.norm(e_x)
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

    # Точка M
    fig.add_trace(go.Scatter3d(
        x=[point_new[0]], y=[point_new[1]], z=[point_new[2]],
        mode='markers',
        marker=dict(color='red', size=8),
        name='Point M'
    ))

    # Векторы ускорений
    vectors = [a_rel_new, a_centr_new, a_rot_new, a_trans_new, a_cor_new, a_abs_new]
    colors = ['blue', 'green', 'orange', 'brown', 'cyan', 'purple']
    labels = ['a_rel', 'a_centr', 'a_rot', 'a_trans_post', 'a_cor', 'a_abs']
    numeric_vals = [np.linalg.norm(v) for v in [a_rel, a_centr, a_rot, a_trans, a_cor, a_abs]]
    for vec, col, lab, val in zip(vectors, colors, labels, numeric_vals):
        fig.add_trace(go.Scatter3d(
            x=[point_new[0], point_new[0]+vec[0]],
            y=[point_new[1], point_new[1]+vec[1]],
            z=[point_new[2], point_new[2]+vec[2]],
            mode='lines+text',
            line=dict(color=col, width=3),
            text=['', f'{lab} = {val:.2f}'],
            textposition='middle right',
            name=lab
        ))

    # Оси новой системы (чёрные)
    axis_len = 2.0
    # Ось X (a_отн⊥)
    fig.add_trace(go.Scatter3d(
        x=[point_new[0], point_new[0]+axis_len],
        y=[point_new[1], point_new[1]],
        z=[point_new[2], point_new[2]],
        mode='lines+text',
        line=dict(color='black', width=2),
        text=['', 'a_отн⊥'],
        textposition='middle right',
        showlegend=False
    ))
    # Ось Y (a_cor)
    fig.add_trace(go.Scatter3d(
        x=[point_new[0], point_new[0]],
        y=[point_new[1], point_new[1]+axis_len],
        z=[point_new[2], point_new[2]],
        mode='lines+text',
        line=dict(color='black', width=2),
        text=['', 'a_cor'],
        textposition='middle right',
        showlegend=False
    ))
    # Ось Z (ω)
    fig.add_trace(go.Scatter3d(
        x=[point_new[0], point_new[0]],
        y=[point_new[1], point_new[1]],
        z=[point_new[2], point_new[2]+axis_len],
        mode='lines+text',
        line=dict(color='black', width=2),
        text=['', 'ω'],
        textposition='middle right',
        showlegend=False
    ))

    fig.update_layout(
        title='Векторы ускорений в точке M (оси: ω, a_cor, a_отн⊥)',
        scene=dict(
            xaxis_title="a_отн⊥",
            yaxis_title="a_cor",
            zaxis_title="ω",
            aspectmode='auto'
        ),
        legend=dict(x=0.8, y=0.9)
    )
    return fig.to_json()
