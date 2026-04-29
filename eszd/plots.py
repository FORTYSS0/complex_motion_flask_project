import numpy as np
import plotly.graph_objects as go


def get_trajectory_points(t_max=2.5, num_points=500):
    """Возвращает точки траектории."""
    t_vals = np.linspace(0.01, t_max, num_points)
    x_t = 8 * np.cos(np.pi * t_vals**2 / 3)
    y_t = 16 * np.sin(np.pi * t_vals**2 / 3)
    return x_t, y_t, t_vals


def get_trajectory_segment(t0=1, num_points=500):
    """Возвращает участок траектории от t=0 до t=t0."""
    t_vals = np.linspace(0, t0, num_points)
    x_t = 8 * np.cos(np.pi * t_vals**2 / 3)
    y_t = 16 * np.sin(np.pi * t_vals**2 / 3)
    return x_t, y_t, t_vals


def draw_axes_2d(fig, origin=(0, 0), length=20, labels=['X', 'Y'], colors=['#333333', '#333333']):
    """Добавляет оси координат на 2D график Plotly с постоянными стрелками."""
    # Ось X
    fig.add_trace(go.Scatter(
        x=[origin[0], origin[0] + length],
        y=[origin[1], origin[1]],
        mode='lines',
        line=dict(color=colors[0], width=2, dash='solid'),
        showlegend=False,
        hoverinfo='none'
    ))
    fig.add_annotation(
        x=origin[0] + length, y=origin[1],
        ax=origin[0] + length - 1.5, ay=origin[1],
        xref='x', yref='y',
        axref='x', ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1.2,
        arrowwidth=2,
        arrowcolor=colors[0]
    )
    fig.add_annotation(
        x=origin[0] + length + 1, y=origin[1],
        text=labels[0],
        showarrow=False,
        font=dict(size=16, color=colors[0], family='Times New Roman')
    )
    
    # Ось Y
    fig.add_trace(go.Scatter(
        x=[origin[0], origin[0]],
        y=[origin[1], origin[1] + length],
        mode='lines',
        line=dict(color=colors[1], width=2, dash='solid'),
        showlegend=False,
        hoverinfo='none'
    ))
    fig.add_annotation(
        x=origin[0], y=origin[1] + length,
        ax=origin[0], ay=origin[1] + length - 1.5,
        xref='x', yref='y',
        axref='x', ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1.2,
        arrowwidth=2,
        arrowcolor=colors[1]
    )
    fig.add_annotation(
        x=origin[0], y=origin[1] + length + 1,
        text=labels[1],
        showarrow=False,
        font=dict(size=16, color=colors[1], family='Times New Roman')
    )
    
    # Точка O
    fig.add_trace(go.Scatter(
        x=[origin[0]],
        y=[origin[1]],
        mode='markers+text',
        marker=dict(color='black', size=8, symbol='circle'),
        text=['<b>O</b>'],
        textposition='bottom right',
        textfont=dict(size=14, color='black', family='Times New Roman'),
        showlegend=False,
        hoverinfo='none'
    ))


def add_vector_2d(fig, start, vector, color, name, show_legend=True):
    """Добавляет вектор со стрелкой на 2D график."""
    end_x = start[0] + vector[0]
    end_y = start[1] + vector[1]
    
    fig.add_trace(go.Scatter(
        x=[start[0], end_x],
        y=[start[1], end_y],
        mode='lines',
        line=dict(color=color, width=4),
        name=name,
        showlegend=show_legend,
        hoverinfo='none'
    ))
    
    angle = np.arctan2(vector[1], vector[0])
    arrow_len = np.linalg.norm(vector) * 0.15
    if arrow_len > 0.1:
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


def eszd_trajectory(data):
    """Рис. 1 - Траектория и положение точки с выделением пройденного пути S."""
    x_t_full, y_t_full, _ = get_trajectory_points()
    x_t_segment, y_t_segment, _ = get_trajectory_segment(t0=data['t'])
    point = data['point']
    s = data['s']
    
    fig = go.Figure()
    draw_axes_2d(fig, length=25)
    
    # Вся траектория (полупрозрачная, для контекста)
    fig.add_trace(go.Scatter(
        x=x_t_full.tolist(),
        y=y_t_full.tolist(),
        mode='lines',
        line=dict(color='#1f77b4', width=2, dash='dot'),
        name='Вся траектория (t > 0)',
        hoverinfo='none'
    ))
    
    # Выделенный путь S от t=0 до t=1 (толстая чёрная линия)
    fig.add_trace(go.Scatter(
        x=x_t_segment.tolist(),
        y=y_t_segment.tolist(),
        mode='lines',
        line=dict(color='black', width=4),
        name=f'<b>Пройденный путь S = {s:.3f} м</b>',
        hoverinfo='none'
    ))
    
    # Стрелка в конце пути для обозначения направления
    if len(x_t_segment) > 1:
        arrow_idx = -min(10, len(x_t_segment) // 10)
        fig.add_annotation(
            x=x_t_segment[arrow_idx],
            y=y_t_segment[arrow_idx],
            ax=x_t_segment[arrow_idx - 1],
            ay=y_t_segment[arrow_idx - 1],
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor='black',
            opacity=0.7
        )
    
    # Подпись S рядом с дугой
    mid_idx = len(x_t_segment) // 2
    fig.add_annotation(
        x=x_t_segment[mid_idx] + 0.5,
        y=y_t_segment[mid_idx] + 0.5,
        text=f'<b>S = {s:.3f} м</b>',
        showarrow=False,
        font=dict(size=14, color='black', family='Times New Roman'),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1,
        borderpad=4
    )
    
    # Точка M в конце пути
    fig.add_trace(go.Scatter(
        x=[point[0]],
        y=[point[1]],
        mode='markers+text',
        marker=dict(color='#d62728', size=14, symbol='circle', line=dict(color='white', width=2)),
        text=['<b>M</b>'],
        textposition='top center',
        textfont=dict(size=14, color='#d62728', family='Times New Roman'),
        name=f'Точка M (t = {data["t"]} с)',
        hovertemplate=f'<b>M</b><br>x = {point[0]:.3f} м<br>y = {point[1]:.3f} м<br>S = {s:.3f} м<extra></extra>'
    ))
    
    # Точка начала движения (t=0)
    x0_t, y0_t, _ = get_trajectory_segment(t0=0.01)
    fig.add_trace(go.Scatter(
        x=[x0_t[0]],
        y=[y0_t[0]],
        mode='markers+text',
        marker=dict(color='green', size=10, symbol='circle', line=dict(color='white', width=1.5)),
        text=['<b>t=0</b>'],
        textposition='bottom right',
        textfont=dict(size=11, color='green', family='Times New Roman'),
        name='Начало движения (t=0)',
        showlegend=True
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>Рис. 1 - Траектория движения точки</b>',
            font=dict(size=18, family='Times New Roman'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='<b>x, м</b>', font=dict(size=14, family='Times New Roman')),
            autorange=True,
            scaleanchor='y',
            scaleratio=1,
            gridcolor='#e0e0e0',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text='<b>y, м</b>', font=dict(size=14, family='Times New Roman')),
            autorange=True,
            gridcolor='#e0e0e0',
            showgrid=True,
            zeroline=False
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(size=11, family='Times New Roman'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#cccccc',
            borderwidth=1
        ),
        plot_bgcolor='white',
        hovermode='closest',
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig.to_json()


def eszd_velocities(data):
    """Рис. 2 - Вектор скорости и касательное ускорение."""
    x_t_full, y_t_full, _ = get_trajectory_points()
    x_t_segment, y_t_segment, _ = get_trajectory_segment(t0=data['t'])
    point = data['point']
    V = data['V']
    V_vec = data['V_vec']
    a_tau = data['a_tau']
    a_tau_vec = data['a_tau_vec']
    s = data['s']
    
    fig = go.Figure()
    draw_axes_2d(fig, length=25)
    
    # Вся траектория
    fig.add_trace(go.Scatter(
        x=x_t_full.tolist(),
        y=y_t_full.tolist(),
        mode='lines',
        line=dict(color='#1f77b4', width=2, dash='dot'),
        name='Вся траектория',
        hoverinfo='none'
    ))
    
    # Выделенный путь S
    fig.add_trace(go.Scatter(
        x=x_t_segment.tolist(),
        y=y_t_segment.tolist(),
        mode='lines',
        line=dict(color='black', width=3),
        name=f'<b>S = {s:.3f} м</b>',
        hoverinfo='none'
    ))
    
    # Точка M
    fig.add_trace(go.Scatter(
        x=[point[0]],
        y=[point[1]],
        mode='markers',
        marker=dict(color='#d62728', size=12, symbol='circle', line=dict(color='white', width=2)),
        name='Точка M',
        hovertemplate=f'<b>M</b><br>x = {point[0]:.3f} м<br>y = {point[1]:.3f} м<br>S = {s:.3f} м<extra></extra>'
    ))
    
    # Вектор скорости
    add_vector_2d(fig, (point[0], point[1]), V_vec, '#2ca02c',
                  f'<b>V</b> = {V:.3f} м/с',
                  show_legend=True)
    
    # Касательное ускорение
    if abs(a_tau) > 0.01:
        add_vector_2d(fig, (point[0], point[1]), a_tau_vec, '#e67e22',
                      f'<b>a_τ</b> = {a_tau:.3f} м/с²',
                      show_legend=True)
    
    fig.update_layout(
        title=dict(
            text='<b>Рис. 2 - Вектор скорости и касательное ускорение</b>',
            font=dict(size=18, family='Times New Roman'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='<b>x, м</b>', font=dict(size=14, family='Times New Roman')),
            autorange=True,
            scaleanchor='y',
            scaleratio=1,
            gridcolor='#e0e0e0',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text='<b>y, м</b>', font=dict(size=14, family='Times New Roman')),
            autorange=True,
            gridcolor='#e0e0e0',
            showgrid=True,
            zeroline=False
        ),
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.98,
            xanchor='left',
            x=0.02,
            font=dict(size=11, family='Times New Roman'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#cccccc',
            borderwidth=1
        ),
        plot_bgcolor='white',
        hovermode='closest',
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig.to_json()


def eszd_accelerations(data):
    """Рис. 3 - Касательное и нормальное ускорения."""
    x_t_full, y_t_full, _ = get_trajectory_points()
    x_t_segment, y_t_segment, _ = get_trajectory_segment(t0=data['t'])
    point = data['point']
    V = data['V']
    a_tau = data['a_tau']
    a_n = data['a_n']
    a_mod = data['a_mod']
    a_tau_vec = data['a_tau_vec']
    a_n_vec = data['a_n_vec']
    rho = data['rho']
    s = data['s']
    
    fig = go.Figure()
    draw_axes_2d(fig, length=25)
    
    # Вся траектория
    fig.add_trace(go.Scatter(
        x=x_t_full.tolist(),
        y=y_t_full.tolist(),
        mode='lines',
        line=dict(color='#1f77b4', width=2, dash='dot'),
        name='Вся траектория',
        hoverinfo='none'
    ))
    
    # Выделенный путь S
    fig.add_trace(go.Scatter(
        x=x_t_segment.tolist(),
        y=y_t_segment.tolist(),
        mode='lines',
        line=dict(color='black', width=3),
        name=f'<b>S = {s:.3f} м</b>',
        hoverinfo='none'
    ))
    
    # Точка M
    fig.add_trace(go.Scatter(
        x=[point[0]],
        y=[point[1]],
        mode='markers',
        marker=dict(color='#d62728', size=12, symbol='circle', line=dict(color='white', width=2)),
        name='Точка M',
        hovertemplate=f'<b>M</b><br>x = {point[0]:.3f} м<br>y = {point[1]:.3f} м<br>S = {s:.3f} м<br>ρ = {rho:.3f} м<extra></extra>'
    ))
    
    # Касательное ускорение
    if abs(a_tau) > 0.01:
        add_vector_2d(fig, (point[0], point[1]), a_tau_vec, '#e67e22',
                      f'<b>a_τ</b> = {a_tau:.3f} м/с²',
                      show_legend=True)
    
    # Нормальное ускорение
    if abs(a_n) > 0.01:
        add_vector_2d(fig, (point[0], point[1]), a_n_vec, '#1f77b4',
                      f'<b>a_n</b> = {a_n:.3f} м/с²  (V²/ρ = {V:.3f}²/{rho:.3f})',
                      show_legend=True)
    
    # Полное ускорение
    add_vector_2d(fig, (point[0], point[1]), a_tau_vec + a_n_vec, '#2ca02c',
                  f'<b>a</b> = {a_mod:.3f} м/с²',
                  show_legend=True)
    
    fig.update_layout(
        title=dict(
            text='<b>Рис. 3 - Касательное и нормальное ускорения</b>',
            font=dict(size=18, family='Times New Roman'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='<b>x, м</b>', font=dict(size=14, family='Times New Roman')),
            autorange=True,
            scaleanchor='y',
            scaleratio=1,
            gridcolor='#e0e0e0',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text='<b>y, м</b>', font=dict(size=14, family='Times New Roman')),
            autorange=True,
            gridcolor='#e0e0e0',
            showgrid=True,
            zeroline=False
        ),
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.98,
            xanchor='left',
            x=0.02,
            font=dict(size=10, family='Times New Roman'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#cccccc',
            borderwidth=1
        ),
        plot_bgcolor='white',
        hovermode='closest',
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig.to_json()
