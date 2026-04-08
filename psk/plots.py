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
    # Стрелка X
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
    # Стрелка Y
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


def add_vector_2d(fig, start, vector, color, name, show_legend=True):
    """Добавляет вектор со стрелкой на 2D график."""
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


def psk_trajectory(data):
    """Рис. 1 — Траектория и положение точки в полярных координатах."""
    x_t, y_t, _ = get_trajectory_points()
    point = data['point']
    r = data['r']
    phi = data['phi']
    
    fig = go.Figure()
    draw_axes_2d(fig, length=25)
    
    # Траектория
    fig.add_trace(go.Scatter(
        x=x_t.tolist(),
        y=y_t.tolist(),
        mode='lines',
        line=dict(color='#1f77b4', width=3),
        name='Траектория движения',
        hoverinfo='none'
    ))
    
    # Радиус-вектор (из O в M)
    fig.add_trace(go.Scatter(
        x=[0, point[0]],
        y=[0, point[1]],
        mode='lines',
        line=dict(color='#9467bd', width=2.5, dash='dot'),
        name=f'<b>r</b> = {r:.3f} м',
        showlegend=True,
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
        hovertemplate=f'<b>M</b><br>r = {r:.3f} м<br>φ = {phi:.3f} рад ({phi*180/np.pi:.1f}°)<br>x = {point[0]:.3f} м<br>y = {point[1]:.3f} м<extra></extra>'
    ))
    
    # Дуга для обозначения угла φ
    theta = np.linspace(0, phi, 50)
    arc_r = r * 0.3
    arc_x = arc_r * np.cos(theta)
    arc_y = arc_r * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=arc_x.tolist(),
        y=arc_y.tolist(),
        mode='lines',
        line=dict(color='#9467bd', width=2, dash='dot'),
        name=f'Угол φ = {phi:.3f} рад',
        showlegend=True,
        hoverinfo='none'
    ))
    
    # Стрелка для обозначения направления угла
    mid_angle = phi / 2
    arrow_r = arc_r * 0.7
    fig.add_annotation(
        x=arrow_r * np.cos(mid_angle),
        y=arrow_r * np.sin(mid_angle),
        text='φ',
        showarrow=False,
        font=dict(size=14, color='#9467bd', family='Times New Roman')
    )
    
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


def psk_velocities(data):
    """Рис. 2 — Радиальная и трансверсальная составляющие скорости."""
    x_t, y_t, _ = get_trajectory_points()
    point = data['point']
    r = data['r']
    phi = data['phi']
    Vr = data['Vr']
    Vphi = data['Vphi']
    V_mod = data['V_mod']
    dr_dt = data['dr_dt']
    dphi_dt = data['dphi_dt']
    
    # Вычисляем единичные векторы в точке M
    e_r = np.array([point[0], point[1]]) / r if r > 0 else np.array([1, 0])
    e_phi = np.array([-e_r[1], e_r[0]])  # поворот на 90° против часовой
    
    # Векторы компонент скорости
    Vr_vec = Vr * e_r
    Vphi_vec = Vphi * e_phi
    
    fig = go.Figure()
    draw_axes_2d(fig, length=25)
    
    # Траектория (полупрозрачная)
    fig.add_trace(go.Scatter(
        x=x_t.tolist(),
        y=y_t.tolist(),
        mode='lines',
        line=dict(color='#1f77b4', width=2, dash='dot'),
        name='Траектория',
        hoverinfo='none'
    ))
    
    # Радиус-вектор
    fig.add_trace(go.Scatter(
        x=[0, point[0]],
        y=[0, point[1]],
        mode='lines',
        line=dict(color='#9467bd', width=2, dash='dot'),
        name=f'r = {r:.3f} м',
        showlegend=True,
        hoverinfo='none'
    ))
    
    # Точка M
    fig.add_trace(go.Scatter(
        x=[point[0]],
        y=[point[1]],
        mode='markers',
        marker=dict(color='#d62728', size=12, symbol='circle', line=dict(color='white', width=2)),
        name='Точка M',
        hovertemplate=f'<b>M</b><br>r = {r:.3f} м<br>φ = {phi:.3f} рад<extra></extra>'
    ))
    
    # Радиальная составляющая скорости Vr (вдоль e_r)
    add_vector_2d(fig, (point[0], point[1]), Vr_vec, '#1f77b4',
                  f'<b>Vr</b> = {Vr:.3f} м/с  (dr/dt = {dr_dt:.3f} м/с)',
                  show_legend=True)
    
    # Трансверсальная составляющая скорости Vφ (вдоль e_φ)
    add_vector_2d(fig, (point[0], point[1]), Vphi_vec, '#e67e22',
                  f'<b>Vφ</b> = {Vphi:.3f} м/с  (r·dφ/dt = {r:.3f}·{dphi_dt:.3f})',
                  show_legend=True)
    
    # Пунктирные линии для правила параллелограмма
    end_Vr = point + Vr_vec
    end_Vphi = point + Vphi_vec
    end_V = point + Vr_vec + Vphi_vec
    
    fig.add_trace(go.Scatter(
        x=[end_Vr[0], end_V[0]],
        y=[end_Vr[1], end_V[1]],
        mode='lines',
        line=dict(color='gray', width=2, dash='dash'),
        showlegend=False,
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(
        x=[end_Vphi[0], end_V[0]],
        y=[end_Vphi[1], end_V[1]],
        mode='lines',
        line=dict(color='gray', width=2, dash='dash'),
        showlegend=False,
        hoverinfo='none'
    ))
    
    # Полная скорость
    add_vector_2d(fig, (point[0], point[1]), Vr_vec + Vphi_vec, '#2ca02c',
                  f'<b>V</b> = {V_mod:.3f} м/с',
                  show_legend=True)
    
    fig.update_layout(
        title=dict(
            text='<b>Рис. 2 — Радиальная и трансверсальная составляющие скорости</b>',
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


def psk_accelerations(data):
    """Рис. 3 — Радиальная и трансверсальная составляющие ускорения."""
    x_t, y_t, _ = get_trajectory_points()
    point = data['point']
    r = data['r']
    phi = data['phi']
    ar = data['ar']
    aphi = data['aphi']
    a_mod = data['a_mod']
    d2r_dt2 = data['d2r_dt2']
    d2phi_dt2 = data['d2phi_dt2']
    dphi_dt = data['dphi_dt']
    dr_dt = data['dr_dt']
    
    # Вычисляем единичные векторы
    e_r = np.array([point[0], point[1]]) / r if r > 0 else np.array([1, 0])
    e_phi = np.array([-e_r[1], e_r[0]])
    
    # Векторы компонент ускорения
    ar_vec = ar * e_r
    aphi_vec = aphi * e_phi
    
    fig = go.Figure()
    draw_axes_2d(fig, length=25)
    
    # Траектория
    fig.add_trace(go.Scatter(
        x=x_t.tolist(),
        y=y_t.tolist(),
        mode='lines',
        line=dict(color='#1f77b4', width=2, dash='dot'),
        name='Траектория',
        hoverinfo='none'
    ))
    
    # Радиус-вектор
    fig.add_trace(go.Scatter(
        x=[0, point[0]],
        y=[0, point[1]],
        mode='lines',
        line=dict(color='#9467bd', width=2, dash='dot'),
        name=f'r = {r:.3f} м',
        showlegend=True,
        hoverinfo='none'
    ))
    
    # Точка M
    fig.add_trace(go.Scatter(
        x=[point[0]],
        y=[point[1]],
        mode='markers',
        marker=dict(color='#d62728', size=12, symbol='circle', line=dict(color='white', width=2)),
        name='Точка M',
        hovertemplate=f'<b>M</b><br>r = {r:.3f} м<br>φ = {phi:.3f} рад<extra></extra>'
    ))
    
    # Радиальная составляющая ускорения ar
    add_vector_2d(fig, (point[0], point[1]), ar_vec, '#1f77b4',
                  f'<b>ar</b> = {ar:.3f} м/с²  (d²r/dt² - r·ω² = {d2r_dt2:.3f} - {r:.3f}·{dphi_dt:.3f}² = {ar:.3f})',
                  show_legend=True)
    
    # Трансверсальная составляющая ускорения aφ
    add_vector_2d(fig, (point[0], point[1]), aphi_vec, '#e67e22',
                  f'<b>aφ</b> = {aphi:.3f} м/с²  (r·ε + 2·Vr·ω = {r:.3f}·{d2phi_dt2:.3f} + 2·{dr_dt:.3f}·{dphi_dt:.3f} = {aphi:.3f})',
                  show_legend=True)
    
    # Пунктирные линии
    end_ar = point + ar_vec
    end_aphi = point + aphi_vec
    end_a = point + ar_vec + aphi_vec
    
    fig.add_trace(go.Scatter(
        x=[end_ar[0], end_a[0]],
        y=[end_ar[1], end_a[1]],
        mode='lines',
        line=dict(color='gray', width=2, dash='dash'),
        showlegend=False,
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(
        x=[end_aphi[0], end_a[0]],
        y=[end_aphi[1], end_a[1]],
        mode='lines',
        line=dict(color='gray', width=2, dash='dash'),
        showlegend=False,
        hoverinfo='none'
    ))
    
    # Полное ускорение
    add_vector_2d(fig, (point[0], point[1]), ar_vec + aphi_vec, '#2ca02c',
                  f'<b>a</b> = {a_mod:.3f} м/с²',
                  show_legend=True)
    
    fig.update_layout(
        title=dict(
            text='<b>Рис. 3 — Радиальная и трансверсальная составляющие ускорения</b>',
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
            font=dict(size=9, family='Times New Roman'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#cccccc',
            borderwidth=1
        ),
        plot_bgcolor='white',
        hovermode='closest',
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig.to_json()
