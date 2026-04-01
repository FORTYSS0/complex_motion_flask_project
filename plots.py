def generate_all_plots(data):
    # ... (первая часть без изменений) ...

    # ---- 2. Скорости (подвижная система) ----
    point = data['point']
    vectors = [data['V_rel'], data['V_rot'], data['V_trans_post'], data['V_abs']]
    colors = ['blue', 'green', 'orange', 'purple']
    labels = ['V_rel', 'V_rot', 'V_trans_post', 'V_abs']
    numeric_vals = [data['V_rel_mod'], data['V_rot_mod'],
                    data['V_trans_post_mod'], data['V_abs_mod']]
    
    # Создаём отдельный график для скоростей, чтобы добавить ω
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Векторы скоростей в точке M')
    
    # Точка M
    ax.scatter(point[0], point[1], point[2], color='red', s=50, label='Point M')
    
    # Векторы скоростей
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
    
    # Добавляем вектор угловой скорости ω из точки O
    O = np.array([0.0, 0.0, data['zp']])
    omega_vec = np.array([0.0, 0.0, data['omega']])
    # Масштабируем для наглядности (длина = 0.3 от размаха данных)
    all_points_for_omega = [np.array(point)] + [np.array(point) + v for v in vectors] + [O, O + omega_vec]
    all_arr = np.array(all_points_for_omega)
    max_range = np.max(np.max(all_arr, axis=0) - np.min(all_arr, axis=0))
    omega_length = max_range * 0.2
    omega_unit = omega_vec / (np.linalg.norm(omega_vec) + 1e-8)
    omega_scaled = omega_unit * omega_length
    ax.quiver(O[0], O[1], O[2],
              omega_scaled[0], omega_scaled[1], omega_scaled[2],
              color='cyan', label='ω', arrow_length_ratio=0.05)
    end_omega = O + omega_scaled
    ax.text(end_omega[0], end_omega[1], end_omega[2],
            f'ω = {data["omega"]:.2f}', color='cyan', fontsize=8)
    
    # Расчёт лимитов (с учётом всех векторов и ω)
    all_points = [np.array(point)] + [np.array(point) + v for v in vectors] + [O, O + omega_scaled] + [np.array([0,0,0])]
    all_arr = np.array(all_points)
    min_vals = np.min(all_arr, axis=0)
    max_vals = np.max(all_arr, axis=0)
    axis_length = max_range * 0.25
    axis_ends = [[axis_length,0,0], [0,axis_length,0], [0,0,axis_length]]
    all_arr = np.vstack([all_arr, axis_ends])
    min_vals = np.min(all_arr, axis=0)
    max_vals = np.max(all_arr, axis=0)
    margin = 0.2
    ax.set_xlim([min_vals[0]-margin, max_vals[0]+margin])
    ax.set_ylim([min_vals[1]-margin, max_vals[1]+margin])
    ax.set_zlim([min_vals[2]-margin, max_vals[2]+margin])
    
    # Оси (подвижная система)
    draw_axes(ax, origin=(0,0,0), length=axis_length, color='black')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'velocities.png'), dpi=150)
    plt.close()
    
    # ---- 3. Ускорения (подвижная система) ----
    # Аналогично добавляем ω и уже имеющийся a_кор
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
    
    # Добавляем ω из точки O
    O = np.array([0.0, 0.0, data['zp']])
    omega_scaled = omega_unit * omega_length  # используем ту же длину, что и для скоростей
    ax.quiver(O[0], O[1], O[2],
              omega_scaled[0], omega_scaled[1], omega_scaled[2],
              color='magenta', label='ω', arrow_length_ratio=0.05)
    end_omega = O + omega_scaled
    ax.text(end_omega[0], end_omega[1], end_omega[2],
            f'ω = {data["omega"]:.2f}', color='magenta', fontsize=8)
    
    # Расчёт лимитов (включая ω)
    all_points = [np.array(point)] + [np.array(point) + v for v in vectors] + [O, O + omega_scaled] + [np.array([0,0,0])]
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
