def generate_interactive_trajectory(data):
    """Возвращает JSON для интерактивного графика траектории с одной стрелкой направления в конце,
       увеличенной стрелкой ω и горизонтальной проекцией траектории на плоскость z = z_M."""
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

    # --- Горизонтальная проекция траектории (плоскость, параллельная X'Y', через точку M) ---
    z_const = data['point'][2]
    fig.add_trace(go.Scatter3d(
        x=x_t, y=y_t, z=np.full_like(z_t, z_const),
        mode='lines',
        line=dict(color='orange', width=2, dash='dash'),
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

    # --- Дуга вращения (ω) с увеличенной стрелкой (конусом) в конце ---
    theta = np.linspace(0, -np.pi/2, 50)
    radius = axis_len * 0.6
    x_arc = O[0] + radius * np.cos(theta)
    y_arc = O[1] + radius * np.sin(theta)
    z_arc = O[2] + axis_len * 0.05
    fig.add_trace(go.Scatter3d(x=x_arc, y=y_arc, z=np.full_like(x_arc, z_arc), mode='lines',
                               line=dict(color='green', width=2), name='ω'))
    # Стрелка (конус) в конце дуги – увеличенный размер
    end_x_arc = x_arc[-1]
    end_y_arc = y_arc[-1]
    end_z_arc = z_arc
    # Вектор направления дуги в последней точке (касательная)
    dx_arc = x_arc[-1] - x_arc[-2]
    dy_arc = y_arc[-1] - y_arc[-2]
    dz_arc = 0
    len_arc = np.sqrt(dx_arc**2 + dy_arc**2 + dz_arc**2)
    if len_arc > 1e-8:
        # Увеличиваем масштаб стрелки
        arrow_scale_arc = 0.3 * radius  # больше, чем для обычных векторов
        u_arc = dx_arc / len_arc * arrow_scale_arc
        v_arc = dy_arc / len_arc * arrow_scale_arc
        w_arc = 0
        fig.add_trace(go.Cone(
            x=[end_x_arc], y=[end_y_arc], z=[end_z_arc],
            u=[u_arc], v=[v_arc], w=[w_arc],
            colorscale=[[0, 'green'], [1, 'green']],
            showscale=False,
            sizemode='scaled',
            sizeref=0.3,  # меньший sizeref делает конус крупнее
            name='ω direction'
        ))
    # Подпись ω
    mid = len(theta)//2
    fig.add_trace(go.Scatter3d(x=[x_arc[mid]], y=[y_arc[mid]], z=[z_arc + 0.02], mode='text',
                               text=[f'ω = {data["omega"]:.2f}'], textfont=dict(color='green', size=10), showlegend=False))

    fig.update_layout(title='Способ задания движения (абсолютная траектория)',
                      scene=dict(xaxis_title="X'", yaxis_title="Y'", zaxis_title="Z'", aspectmode='auto'),
                      legend=dict(x=0.8, y=0.9))
    return fig.to_json()
