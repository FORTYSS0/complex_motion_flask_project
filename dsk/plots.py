import numpy as np
import plotly.graph_objects as go


def get_trajectory_points(t_max=2.5, num_points=500):
    """Возвращает точки траектории."""
    t_vals = np.linspace(0.01, t_max, num_points)
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
    # Стрелка X (всегда видима)
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
    # Подпись X
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
    # Стрелка Y (всегда видима)
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
    # Подпись Y
    fig.add_annotation(
        x=origin[0], y=origin[1] + length + 1,
        text=labels[1],
        showarrow=False,
        font=dict(size=16, color=colors[1], family='Times New Roman')
    )
    
    # Точка O (центр координат)
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


def add_vector_component_2d(fig, start, component, color, name, axis_name, is_dashed=False):
    """Добавляет компоненту вектора вдоль оси и опционально пунктирную линию."""
    end_x = start[0] + component[0]
    end_y = start[1] + component[1]
    
    # Линия компоненты
    line_style = dict(color=color, width=3)
    if is_dashed:
        line_style['dash'] = 'dot'
    
    fig.add_trace(go.Scatter(
        x=[start[0], end_x],
        y=[start[1], end_y],
        mode='lines',
        line=line_style,
        name=name,
        showlegend=True,
        hoverinfo='none'
    ))
    
    # Стрелка на конце компоненты
    angle = np.arctan2(component[1], component[0])
    arrow_len = np.linalg.norm(component) * 0.15
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


def add_vector_2d(fig, start, vector, color, name, show_legend=True):
    """Добавляет результирующий вектор со стрелкой."""
    end_x = start[0] + vector[0]
    end_y = start[1] + vector[1]
    
    # Линия вектора
    fig.add_trace(go.Scatter(
        x=[start[0], end_x],
        y=[start[1], end_y],
        mode='lines',
        line=dict(color=color, width=4),
        name=name,
        showlegend=show_legend,
        hoverinfo='none'
    ))
    
    # Стрелка
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


def dsk_trajectory(data):
    """Рис. 1 — Траектория и положение точки."""
    x_t, y_t, _ = get_trajectory_points()
    point = data['point']
    
    fig = go.Figure()
    draw_axes_2d(fig, length=20)
    draw_axes_2d(fig, length=1, labels=['i', 'j'], colors=['red', 'green'])
    
    # Траектория
    fig.add_trace(go.Scatter(
        x=x_t.tolist(),
        y=y_t.tolist(),
        mode='lines',
        line=dict(color='#1f77b4', width=3),
        name='Траектория движения',
        hoverinfo='none'
    ))
    
    # Точка M
    fig.add_trace(go.Scatter(
        x=[point[0]],
        y=[point[1]],
        mode='markers+text',
        marker=dict(color='#d62728', size=14, symbol='circle', line=dict(color='white', width=2)),
        text=['<b>M</b>'],
        textposition='top center',
        textfont=dict(size=14, color='#d62728', family='Times New Roman'),
        name=f'Точка M (t = {data["t"]} с)',
        hovertemplate=f'<b>M</b><br>x = {point[0]:.3f} м<br>y = {point[1]:.3f} м<extra></extra>'
    ))
    
    # Настройка оформления
    fig.update_layout(
        title=dict(
            text='<b>Рис. 1 — Траектория движения точки</b>',
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
            font=dict(size=12, family='Times New Roman'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#cccccc',
            borderwidth=1
        ),
        plot_bgcolor='white',
        hovermode='closest',
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig.to_json()


def dsk_velocities(data):
    """Рис. 2 — Вектор скорости и его компоненты."""
    x_t, y_t, _ = get_trajectory_points()
    point = data['point']
    V = data['V']
    vx, vy = V[0], V[1]
    V_mod = data['V_mod']
    
    fig = go.Figure()
    draw_axes_2d(fig, length=25)
    draw_axes_2d(fig, length=1, labels=['i', 'j'], colors=['red', 'green'])
    
    # Траектория (полупрозрачная)
    fig.add_trace(go.Scatter(
        x=x_t.tolist(),
        y=y_t.tolist(),
        mode='lines',
        line=dict(color='#1f77b4', width=2, dash='dot'),
        name='Траектория',
        hoverinfo='none'
    ))
    
    # Точка M
    fig.add_trace(go.Scatter(
        x=[point[0]],
        y=[point[1]],
        mode='markers',
        marker=dict(color='#d62728', size=12, symbol='circle', line=dict(color='white', width=2)),
        name='Точка M',
        hovertemplate=f'<b>M</b><br>x = {point[0]:.3f} м<br>y = {point[1]:.3f} м<extra></extra>'
    ))
    
    # Компоненты скорости
    # Vx (по оси X)
    add_vector_component_2d(fig, (point[0], point[1]), (vx, 0), '#1f77b4', f'<b>Vx</b> = {vx:.3f} м/с', 'X')
    # Vy (по оси Y)
    add_vector_component_2d(fig, (point[0], point[1]), (0, vy), "#b41f53", f'<b>Vy</b> = {vy:.3f} м/с', 'Y')
    
    # Пунктирные линии для правила параллелограмма
    # От конца Vx к концу V
    fig.add_trace(go.Scatter(
        x=[point[0] + vx, point[0] + vx],
        y=[point[1], point[1] + vy],
        mode='lines',
        line=dict(color='gray', width=2, dash='dash'),
        showlegend=False,
        hoverinfo='none'
    ))
    # От конца Vy к концу V
    fig.add_trace(go.Scatter(
        x=[point[0], point[0] + vx],
        y=[point[1] + vy, point[1] + vy],
        mode='lines',
        line=dict(color='gray', width=2, dash='dash'),
        showlegend=False,
        hoverinfo='none'
    ))
    
    # Результирующий вектор скорости
    add_vector_2d(fig, (point[0], point[1]), (vx, vy), '#e67e22',
                  f'<b>V</b> = ({vx:.3f}, {vy:.3f}) м/с<br>|V| = {V_mod:.3f} м/с',
                  show_legend=True)
    
    fig.update_layout(
        title=dict(
            text='<b>Рис. 2 — Вектор скорости точки M',
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


def dsk_accelerations(data):
    """Рис. 3 — Вектор ускорения и его компоненты."""
    x_t, y_t, _ = get_trajectory_points()
    point = data['point']
    a = data['a']
    ax, ay = a[0], a[1]
    a_mod = data['a_mod']
    
    fig = go.Figure()
    draw_axes_2d(fig, length=25)
    draw_axes_2d(fig, length=1, labels=['i', 'j'], colors=['red', 'green'])
    
    # Траектория (полупрозрачная)
    fig.add_trace(go.Scatter(
        x=x_t.tolist(),
        y=y_t.tolist(),
        mode='lines',
        line=dict(color='#1f77b4', width=2, dash='dot'),
        name='Траектория',
        hoverinfo='none'
    ))
    
    # Точка M
    fig.add_trace(go.Scatter(
        x=[point[0]],
        y=[point[1]],
        mode='markers',
        marker=dict(color='#d62728', size=12, symbol='circle', line=dict(color='white', width=2)),
        name='Точка M',
        hovertemplate=f'<b>M</b><br>x = {point[0]:.3f} м<br>y = {point[1]:.3f} м<extra></extra>'
    ))
    
    # Компоненты ускорения
    # ax (по оси X)
    add_vector_component_2d(fig, (point[0], point[1]), (ax, 0), '#1f77b4', f'<b>ax</b> = {ax:.3f} м/с²', 'X')
    # ay (по оси Y)
    add_vector_component_2d(fig, (point[0], point[1]), (0, ay), '#b41f53', f'<b>ay</b> = {ay:.3f} м/с²', 'Y')
    
    # Пунктирные линии для правила параллелограмма
    fig.add_trace(go.Scatter(
        x=[point[0] + ax, point[0] + ax],
        y=[point[1], point[1] + ay],
        mode='lines',
        line=dict(color='gray', width=2, dash='dash'),
        showlegend=False,
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(
        x=[point[0], point[0] + ax],
        y=[point[1] + ay, point[1] + ay],
        mode='lines',
        line=dict(color='gray', width=2, dash='dash'),
        showlegend=False,
        hoverinfo='none'
    ))
    
    # Результирующий вектор ускорения
    add_vector_2d(fig, (point[0], point[1]), (ax, ay), '#2ca02c',
                  f'<b>a</b> = ({ax:.3f}, {ay:.3f}) м/с²<br>|a| = {a_mod:.3f} м/с²',
                  show_legend=True)
    
    fig.update_layout(
        title=dict(
            text='<b>Рис. 3 — Вектор ускорения точки M</b>',
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
