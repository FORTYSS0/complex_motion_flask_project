import numpy as np
import plotly.graph_objects as go


def get_trajectory_points(t_max=1.8, num_points=300):
    """Возвращает точки траектории."""
    t_vals = np.linspace(0.01, t_max, num_points)
    x_t = 8 * np.cos(np.pi * t_vals**2 / 3)
    y_t = 16 * np.sin(np.pi * t_vals**2 / 3)
    return x_t, y_t, t_vals


def draw_axes_2d(fig, origin=(0, 0), length=30, labels=['X', 'Y'], colors=['black', 'black']):
    """Добавляет оси координат на 2D график."""
    # Ось X
    fig.add_trace(go.Scatter(
        x=[origin[0], origin[0] + length],
        y=[origin[1], origin[1]],
        mode='lines+text',
        line=dict(color=colors[0], width=2),
        text=['', labels[0]],
        textposition='top right',
        textfont=dict(size=14, color=colors[0]),
        showlegend=False,
        hoverinfo='none'
    ))
    # Ось Y
    fig.add_trace(go.Scatter(
        x=[origin[0], origin[0]],
        y=[origin[1], origin[1] + length],
        mode='lines+text',
        line=dict(color=colors[1], width=2),
        text=['', labels[1]],
        textposition='top center',
        textfont=dict(size=14, color=colors[1]),
        showlegend=False,
        hoverinfo='none'
    ))


def add_vector_2d(fig, start, vector, color, name, scale=1.0):
    """Добавляет вектор со стрелкой на 2D график."""
    end_x = start[0] + vector[0] * scale
    end_y = start[1] + vector[1] * scale
    
    # Линия вектора
    fig.add_trace(go.Scatter(
        x=[start[0], end_x],
        y=[start[1], end_y],
        mode='lines',
        line=dict(color=color, width=3),
        name=name,
        showlegend=True
    ))
    
    # Стрелка (треугольник)
    angle = np.arctan2(vector[1], vector[0])
    arrow_len = np.linalg.norm(vector) * scale * 0.15
    arrow_angle = np.pi / 6
    
    arrow_x = [
        end_x,
        end_x - arrow_len * np.cos(angle - arrow_angle),
        end_x - arrow_len * np.cos(angle + arrow_angle)
    ]
    arrow_y = [
        end_y,
        end_y - arrow_len * np.sin(angle - arrow_angle),
        end_y - arrow_len * np.sin(angle + arrow_angle)
    ]
    
    fig.add_trace(go.Scatter(
        x=arrow_x, y=arrow_y,
        mode='lines',
        line=dict(color=color, width=2),
        fill='toself',
        fillcolor=color,
        showlegend=False,
        hoverinfo='none'
    ))


def dsk_trajectory(data):
    """Рис. 1 — Траектория и положение точки."""
    x_t, y_t, _ = get_trajectory_points()
    point = data['point']
    
    fig = go.Figure()
    draw_axes_2d(fig, length=35)
    
    # Траектория
    fig.add_trace(go.Scatter(
        x=x_t.tolist(),
        y=y_t.tolist(),
        mode='lines',
        line=dict(color='#1f77b4', width=3),
        name='Траектория точки'
    ))
    
    # Точка M
    fig.add_trace(go.Scatter(
        x=[point[0]],
        y=[point[1]],
        mode='markers+text',
        marker=dict(color='red', size=12, symbol='circle'),
        text=[f'<b>M</b>'],
        textposition='top center',
        textfont=dict(size=12, color='red'),
        name='Положение точки'
    ))
    
    fig.update_layout(
        title=dict(text='<b>Рис. 1 — Траектория движения точки в ДСК</b>', font=dict(size=16)),
        xaxis=dict(title='<b>x, м</b>', range=[-12, 22], scaleanchor='y', scaleratio=1),
        yaxis=dict(title='<b>y, м</b>', range=[-18, 18]),
        width=None,
        height=550,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        plot_bgcolor='white',
        hovermode='closest'
    )
    
    return fig.to_json()


def dsk_velocities(data):
    """Рис. 2 — Вектор скорости в точке M."""
    x_t, y_t, _ = get_trajectory_points()
    point = data['point']
    V = data['V_abs']
    vx, vy = V[0], V[1]
    
    fig = go.Figure()
    draw_axes_2d(fig, length=35)
    
    # Траектория (полупрозрачная)
    fig.add_trace(go.Scatter(
        x=x_t.tolist(),
        y=y_t.tolist(),
        mode='lines',
        line=dict(color='#1f77b4', width=3),
        name='Траектория'
    ))
    
    # Точка M
    fig.add_trace(go.Scatter(
        x=[point[0]],
        y=[point[1]],
        mode='markers',
        marker=dict(color='red', size=10, symbol='circle'),
        name='M',
        showlegend=True
    ))
    
    # Вектор скорости
    add_vector_2d(fig, (point[0], point[1]), (vx, vy), '#e67e22', 
                  f'V = ({vx:.2f}, {vy:.2f}) м/с<br>|V| = {data["V_abs_mod"]:.3f} м/с', 
                  scale=1)
    
    fig.update_layout(
        title=dict(text='<b>Рис. 2 — Вектор скорости в точке M</b>', font=dict(size=16)),
        xaxis=dict(title='<b>x, м</b>', range=[-12, 22], scaleanchor='y', scaleratio=1),
        yaxis=dict(title='<b>y, м</b>', range=[-18, 18]),
        width=None,
        height=550,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        plot_bgcolor='white'
    )
    
    return fig.to_json()


def dsk_accelerations(data):
    """Рис. 3 — Вектор ускорения в точке M."""
    x_t, y_t, _ = get_trajectory_points()
    point = data['point']
    a = data['a_abs']
    ax, ay = a[0], a[1]
    
    fig = go.Figure()
    draw_axes_2d(fig, length=35)
    
    # Траектория (полупрозрачная)
    fig.add_trace(go.Scatter(
        x=x_t.tolist(),
        y=y_t.tolist(),
        mode='lines',
        line=dict(color='#1f77b4', width=3),
        name='Траектория'
    ))
    
    # Точка M
    fig.add_trace(go.Scatter(
        x=[point[0]],
        y=[point[1]],
        mode='markers',
        marker=dict(color='red', size=10, symbol='circle'),
        name='M',
        showlegend=True
    ))
    
    # Вектор ускорения
    add_vector_2d(fig, (point[0], point[1]), (ax, ay), '#d62728',
                  f'a = ({ax:.2f}, {ay:.2f}) м/с²<br>|a| = {data["a_abs_mod"]:.3f} м/с²',
                  scale=1)
    
    fig.update_layout(
        title=dict(text='<b>Рис. 3 — Вектор ускорения в точке M</b>', font=dict(size=16)),
        xaxis=dict(title='<b>x, м</b>', range=[-12, 22], scaleanchor='y', scaleratio=1),
        yaxis=dict(title='<b>y, м</b>', range=[-18, 18]),
        width=None,
        height=550,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        plot_bgcolor='white'
    )
    
    return fig.to_json()
