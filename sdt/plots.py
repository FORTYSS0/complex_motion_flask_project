import numpy as np
import plotly.graph_objects as go


def add_fixed_hatch_3d(fig, origin, axis_dir, perp_dir, length, color='black',
                       n=3, inset=2.5, spacing=1.6, stroke=2.0):
    """Короткие штрихи у конца 3D-оси — признак неподвижной (зафиксированной) оси."""
    d = np.array(axis_dir, dtype=float); d = d / np.linalg.norm(d)
    p = np.array(perp_dir, dtype=float); p = p / np.linalg.norm(p)
    h = (d + p) / np.sqrt(2.0)               # направление штриха (45°)
    tip = np.array(origin, dtype=float) + d * length
    for i in range(n):
        c = tip - d * (inset + i * spacing)
        a = c - 0.5 * stroke * h
        b = c + 0.5 * stroke * h
        fig.add_trace(go.Scatter3d(
            x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False, hoverinfo='none'
        ))


def draw_axes(fig, origin=(0,0,0), length=20, labels=['X', 'Y', 'Z'], colors = ['black', 'black', 'black'], fixed=False):
    """Добавляет оси координат на Plotly график.

    Если fixed=True, на концах осей рисуются штрихи — признак неподвижной
    (зафиксированной) системы отсчёта.
    """
    # Перпендикуляры для штрихов по каждой из осей X, Y, Z
    perps = [(0, 1, 0), (1, 0, 0), (1, 0, 0)]
    for i, (label, color) in enumerate(zip(labels, colors)):
        end = list(origin)
        end[i] += length
        if fixed:
            axis_dir = [1 if j == i else 0 for j in range(3)]
            add_fixed_hatch_3d(fig, origin, axis_dir, perps[i], length, color=color)
        
        # Линия оси
        fig.add_trace(go.Scatter3d(
            x=[origin[0], end[0]],
            y=[origin[1], end[1]],
            z=[origin[2], end[2]],
            mode='lines',
            line=dict(color=color, width=3),
            name=label,
            showlegend=False
        ))
        
        # Стрелка
        cone_size = length * 0.1
        direction = [1 if i==0 else 0, 1 if i==1 else 0, 1 if i==2 else 0]
        fig.add_trace(go.Cone(
            x=[end[0] - direction[0] * cone_size],
            y=[end[1] - direction[1] * cone_size],
            z=[end[2] - direction[2] * cone_size],
            u=[direction[0]], 
            v=[direction[1]], 
            w=[direction[2]],
            colorscale=[[0, color], [1, color]],
            showscale=False,
            sizemode="scaled",
            sizeref=cone_size,
            showlegend=False
        ))
        
        # Подпись
        fig.add_trace(go.Scatter3d(
            x=[end[0] + (0.5 if i==0 else 0)],
            y=[end[1] + (0.5 if i==1 else 0)],
            z=[end[2] + (0.5 if i==2 else 0)],
            mode='text',
            text=[label],
            textfont=dict(size=14, color=color),
            showlegend=False
        ))


def get_trajectory_points(t_max=2.5, num_points=200):
    """Возвращает точки АБСОЛЮТНОЙ траектории для построения.

    Точка движется в подвижной системе xOy (относительное движение),
    а сама подвижная система поворачивается вокруг оси Z на угол
    phi'(t) = 2*sin(pi*t) и смещается вдоль Z по закону z'(t) = 2t^2 + 4.

    Поэтому абсолютные координаты получаются применением матрицы
    перехода (поворота) R_z(phi') к относительному радиус-вектору:

        r_abs(t) = R_z(phi'(t)) * r_rel(t) + (0, 0, z'(t)),

        R_z(phi') = | cos phi'  -sin phi'  0 |
                    | sin phi'   cos phi'  0 |
                    |    0          0      1 |

    Без этого поворота касательная к нарисованной кривой не совпадает
    с вектором абсолютной скорости V_abs (ошибка «скорости по касательной»).
    """
    t_vals = np.linspace(0, t_max, num_points)

    # Относительные координаты (в подвижной системе xOy)
    x_rel = 8 * np.cos(np.pi * t_vals**2 / 3)
    y_rel = 16 * np.sin(np.pi * t_vals**2 / 3)

    # Угол поворота подвижной системы и матрица перехода R_z(phi')
    phi = 2 * np.sin(np.pi * t_vals)
    cos_p, sin_p = np.cos(phi), np.sin(phi)

    # Абсолютные координаты: поворот относительного радиус-вектора
    x_t = cos_p * x_rel - sin_p * y_rel
    y_t = sin_p * x_rel + cos_p * y_rel
    z_t = 2 * t_vals**2 + 4
    return x_t, y_t, z_t, t_vals


def vector_display_scale(vectors, target=18.0):
    """Единый масштаб отображения для группы векторов.

    Самый длинный вектор приводится к длине target (для соразмерности со
    сценой и траекторией), пропорции между векторами сохраняются. Истинные
    модули указываются в подписи легенды (см. add_vector_with_arrow).
    """
    mags = [float(np.linalg.norm(np.asarray(v, dtype=float))) for v in vectors]
    m = max(mags) if mags else 0.0
    return (target / m) if m > 1e-9 else 1.0


def add_vector_with_arrow(fig, start, vector, color, name, scale=1.0):
    """Добавляет вектор со стрелкой на 3D график Plotly.

    Вектор рисуется в масштабе scale (для соразмерности со сценой), а в
    подписи легенды приводится ИСТИННЫЙ модуль вектора.
    """
    start = np.asarray(start, dtype=float)
    vector = np.asarray(vector, dtype=float)
    mag = float(np.linalg.norm(vector))
    label = ('%s = %.1f' % (name, mag)).replace('.', ',')
    end = start + vector * scale

    # Рисуем линию вектора
    fig.add_trace(go.Scatter3d(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        z=[start[2], end[2]],
        mode='lines',
        line=dict(color=color, width=4),
        name=label,
        showlegend=True
    ))

    # Добавляем конус (стрелку) в конце вектора
    # Нормализуем направление
    direction = vector / (np.linalg.norm(vector) + 1e-10)
    # Размер конуса пропорционален ОТОБРАЖАЕМОЙ длине вектора
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

def sdt_trajectory(data):
    """Возвращает JSON для интерактивного графика траектории с эллипсом в момент времени t."""
    x_t, y_t, z_t, _ = get_trajectory_points()
    
    # Координаты точки M в момент t=1
    Mx = float(data['point'][0])
    My = float(data['point'][1])
    Mz = float(data['point'][2])
    
    # Параметры эллипса из уравнений движения
    a_val = 8  # полуось по X
    b_val = 16  # полуось по Y (2 * a = 16)
    
    # Угол в момент t=1: φ = π * t² / 3 = π/3 = 60°
    phi_t = np.pi * (1**2) / 3  # = π/3
    
    # Строим эллипс
    theta = np.linspace(0, 2*np.pi, 200)
    ellipse_x = a_val * np.cos(theta)
    ellipse_y = b_val * np.sin(theta)
    ellipse_z = np.full_like(theta, Mz)
    
    # Точка M на эллипсе должна быть при θ = phi_t
    # x = 8*cos(π/3) = 4, y = 16*sin(π/3) = 13.856
    
    fig = go.Figure()
    draw_axes(fig, length=25, fixed=True)
    draw_axes(fig, length=1, labels=['i', 'j', 'k'], colors=['red', 'green', 'blue'])
    
    # 1. Абсолютная траектория
    fig.add_trace(go.Scatter3d(
        x=x_t.tolist(),
        y=y_t.tolist(),
        z=z_t.tolist(),
        mode='lines',
        line=dict(color='blue', width=4),
        name='Абсолютная траектория'
    ))
    
    # 2. Эллипс (относительное движение) в момент t=1
    fig.add_trace(go.Scatter3d(
        x=ellipse_x.tolist(),
        y=ellipse_y.tolist(),
        z=ellipse_z.tolist(),
        mode='lines',
        line=dict(color='red', width=3, dash='dash'),
        name=f'Эллипс в t=1 (8×16)'
    ))
    
    # 3. Точка M
    fig.add_trace(go.Scatter3d(
        x=[Mx],
        y=[My],
        z=[Mz],
        mode='markers',
        marker=dict(color='red', size=10),
        name=f'M (t=1)'
    ))
    
    fig.update_layout(
        title='Траектория и эллипс относительного движения в момент t=1',
        scene=dict(
            xaxis_title="X'",
            yaxis_title="Y'",
            zaxis_title="Z'",
            aspectmode='data'
        ),
        legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.5)
    )
    return fig.to_json()

def sdt_velocities(data):
    """Интерактивный график скоростей со стрелками."""
    point = data['point']
    
    fig = go.Figure()
    draw_axes(fig, length=20, fixed=True)
    draw_axes(fig, length=1, labels=['i', 'j', 'k'], colors = ['red', 'green', 'blue'])
    
    fig.add_trace(go.Scatter3d(
        x=[point[0]], 
        y=[point[1]], 
        z=[point[2]],
        mode='markers', 
        marker=dict(color='red', size=8), 
        name='Point M'
        )
    )
    
    # Векторы скоростей — в едином масштабе отображения (истинные модули в легенде)
    s = vector_display_scale([data['V_rel'], data['V_rot'],
                              data['V_trans_post'], data['V_abs']], target=18.0)
    add_vector_with_arrow(fig, point, data['V_rel'], 'blue', 'V_rel', scale=s)
    add_vector_with_arrow(fig, point, data['V_rot'], 'green', 'V_rot', scale=s)
    add_vector_with_arrow(fig, point, data['V_trans_post'], 'orange', 'V_trans_post', scale=s)
    add_vector_with_arrow(fig, point, data['V_abs'], 'purple', 'V_abs', scale=s)

    fig.update_layout(
        title='Векторы скоростей в точке M',
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode='data'),
        legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.5),
        margin=dict(l=0, r=0, t=30, b=50)
    )
    return fig.to_json()


def sdt_accelerations(data):
    """Интерактивный график ускорений со стрелками."""
    point = data['point']
    
    fig = go.Figure()
    draw_axes(fig, length=25, fixed=True)
    draw_axes(fig, length=1, labels=['i', 'j', 'k'], colors = ['red', 'green', 'blue'])

    fig.add_trace(go.Scatter3d(
        x=[point[0]],
        y=[point[1]],
        z=[point[2]],
        mode='markers',
        marker=dict(color='red', size=8),
        name='Point M'
        )
    )

    # Векторы ускорений — в едином масштабе отображения (истинные модули в легенде)
    s = vector_display_scale([data['a_rel'], data['a_centr'], data['a_rot'],
                              data['a_trans_post'], data['a_cor'], data['a_abs']], target=20.0)
    add_vector_with_arrow(fig, point, data['a_rel'], 'blue', 'a_rel', scale=s)
    add_vector_with_arrow(fig, point, data['a_centr'], 'green', 'a_centr', scale=s)
    add_vector_with_arrow(fig, point, data['a_rot'], 'orange', 'a_rot', scale=s)
    add_vector_with_arrow(fig, point, data['a_trans_post'], 'brown', 'a_trans_post', scale=s)
    add_vector_with_arrow(fig, point, data['a_cor'], 'cyan', 'a_cor', scale=s)
    add_vector_with_arrow(fig, point, data['a_abs'], 'purple', 'a_abs', scale=s)

    fig.update_layout(
        title='Векторы ускорений в точке M',
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode='data'),
        legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.5),
        margin=dict(l=0, r=0, t=30, b=50)
    )
    return fig.to_json()


def sdt_trajectory_with_velocities(data):
    """Комбинированный график: траектория + скорости со стрелками."""
    x_t, y_t, z_t, _ = get_trajectory_points()
    point = data['point']
    
    fig = go.Figure()
    draw_axes(fig, length=25, fixed=True)
    draw_axes(fig, length=1, labels=['i', 'j', 'k'], colors = ['red', 'green', 'blue'])

    # Добавляем траекторию
    fig.add_trace(go.Scatter3d(
        x=x_t.tolist(),
        y=y_t.tolist(),
        z=z_t.tolist(),
        mode='lines',
        line=dict(color='black', width=4),
        name='Траектория'
    ))

    fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]],
                               mode='markers', marker=dict(color='red', size=8), name='M (t=1)'))

    # Векторы скоростей — в едином масштабе отображения (истинные модули в легенде)
    s = vector_display_scale([data['V_rel'], data['V_rot'],
                              data['V_trans_post'], data['V_abs']], target=20.0)
    add_vector_with_arrow(fig, point, data['V_rel'], 'blue', 'V_rel', scale=s)
    add_vector_with_arrow(fig, point, data['V_rot'], 'green', 'V_rot', scale=s)
    add_vector_with_arrow(fig, point, data['V_trans_post'], 'orange', 'V_trans_post', scale=s)
    add_vector_with_arrow(fig, point, data['V_abs'], 'purple', 'V_abs', scale=s)

    fig.update_layout(
        title='Траектория и векторы скоростей',
        scene=dict(xaxis_title="X'", yaxis_title="Y'", zaxis_title="Z'", aspectmode='data'),
        legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.5),
        margin=dict(l=0, r=0, t=30, b=50)
    )
    return fig.to_json()


def sdt_trajectory_with_accelerations(data):
    """Комбинированный график: траектория + ускорения со стрелками."""
    x_t, y_t, z_t, _ = get_trajectory_points()
    point = data['point']
    
    fig = go.Figure()
    draw_axes(fig, length=25, fixed=True)
    draw_axes(fig, length=1, labels=['i', 'j', 'k'], colors = ['red', 'green', 'blue'])

    # Добавляем траекторию
    fig.add_trace(go.Scatter3d(
        x=x_t.tolist(),
        y=y_t.tolist(),
        z=z_t.tolist(),
        mode='lines',
        line=dict(color='black', width=4),
        name='Траектория'
    ))

    fig.add_trace(go.Scatter3d(
        x=[point[0]],
        y=[point[1]],
        z=[point[2]],
        mode='markers',
        marker=dict(color='red', size=8),
        name='M (t=1)'
        )
    )

    # Векторы ускорений — в едином масштабе отображения (истинные модули в легенде)
    s = vector_display_scale([data['a_rel'], data['a_centr'], data['a_rot'],
                              data['a_trans_post'], data['a_cor'], data['a_abs']], target=20.0)
    add_vector_with_arrow(fig, point, data['a_rel'], 'blue', 'a_rel', scale=s)
    add_vector_with_arrow(fig, point, data['a_centr'], 'green', 'a_centr', scale=s)
    add_vector_with_arrow(fig, point, data['a_rot'], 'orange', 'a_rot', scale=s)
    add_vector_with_arrow(fig, point, data['a_trans_post'], 'brown', 'a_trans_post', scale=s)
    add_vector_with_arrow(fig, point, data['a_cor'], 'cyan', 'a_cor', scale=s)
    add_vector_with_arrow(fig, point, data['a_abs'], 'purple', 'a_abs', scale=s)

    fig.update_layout(
        title='Траектория и векторы ускорений',
        scene=dict(xaxis_title="X'", yaxis_title="Y'", zaxis_title="Z'", aspectmode='data'),
        legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.5),
        margin=dict(l=0, r=0, t=30, b=50)
    )
    return fig.to_json()
