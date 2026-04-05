import numpy as np
import plotly.graph_objects as go


def draw_axes(fig, origin=(0,0,0), length=20, labels=['X', 'Y', 'Z'], colors = ['black', 'black', 'black']):
    """Добавляет оси координат на Plotly график."""
    for i, (label, color) in enumerate(zip(labels, colors)):
        end = list(origin)
        end[i] += length
        
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
    draw_axes(fig, length=25)
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
            aspectmode='auto'
        ),
        legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.5)
    )
    return fig.to_json()

def sdt_velocities(data):
    """Интерактивный график скоростей со стрелками."""
    point = data['point']
    
    fig = go.Figure()
    draw_axes(fig, length=20)
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


def sdt_accelerations(data):
    """Интерактивный график ускорений со стрелками."""
    point = data['point']
    
    fig = go.Figure()
    draw_axes(fig, length=90)
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


def sdt_trajectory_with_velocities(data):
    """Комбинированный график: траектория + скорости со стрелками."""
    x_t, y_t, z_t, _ = get_trajectory_points()
    point = data['point']
    
    fig = go.Figure()
    draw_axes(fig, length=20)
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


def sdt_trajectory_with_accelerations(data):
    """Комбинированный график: траектория + ускорения со стрелками."""
    x_t, y_t, z_t, _ = get_trajectory_points()
    point = data['point']
    
    fig = go.Figure()
    draw_axes(fig, length=90)
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
