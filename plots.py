import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_axes(ax, origin=(0,0,0), length=None, color='black', labels=['X', 'Y', 'Z']):
    """Рисует оси координат заданной длины (length). Длина должна быть определена заранее."""
    for i, label in enumerate(labels):
        vec = [0,0,0]
        vec[i] = length
        ax.quiver(origin[0], origin[1], origin[2],
                  vec[0], vec[1], vec[2],
                  color=color, label=label, arrow_length_ratio=0.03, linewidth=2)
        ax.text(origin[0]+vec[0], origin[1]+vec[1], origin[2]+vec[2], label,
                color=color, fontsize=10, ha='center', va='center')

def plot_vectors(title, point, vectors, colors, labels, filename, numeric_values=None):
    """
    Сохраняет 3D-график с векторами из точки.
    Все векторы и оси гарантированно помещаются в кадре.
    Подписи векторов смещаются перпендикулярно, чтобы не накладываться.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Точка M
    ax.scatter(point[0], point[1], point[2], color='red', s=50, label='Point M')

    # Векторы и подписи
    for idx, (vec, color, label) in enumerate(zip(vectors, colors, labels)):
        ax.quiver(point[0], point[1], point[2],
                  vec[0], vec[1], vec[2],
                  color=color, label=label, arrow_length_ratio=0.03)
        # Подпись с числовым значением, если передано
        if numeric_values and idx < len(numeric_values):
            end = np.array(point) + vec
            # Нормализуем направление вектора
            norm_vec = vec / (np.linalg.norm(vec) + 1e-8)
            # Создаём перпендикулярный вектор в плоскости XY (для простоты)
            # Для 3D лучше использовать вектор, ортогональный к оси Z (чтобы текст не накладывался на линии)
            perp = np.array([-norm_vec[1], norm_vec[0], 0])
            # Если длина перпендикуляра мала (вектор почти вертикален), берём другое направление
            if np.linalg.norm(perp) < 0.1:
                perp = np.array([1, 0, 0])
            perp = perp / np.linalg.norm(perp) * 0.2  # Смещение
            text_pos = end + perp
            ax.text(text_pos[0], text_pos[1], text_pos[2],
                    f'{label} = {numeric_values[idx]:.2f}',
                    color=color, fontsize=8, ha='center', va='center')

    # Собираем все точки для расчёта лимитов: начало (0,0,0), точка M, концы всех векторов
    all_points = [np.array(point)]
    for vec in vectors:
        all_points.append(np.array(point) + vec)
    all_points.append(np.array([0,0,0]))

    # Добавляем также позиции подписей (с запасом)
    # Для простоты добавим увеличенные концы векторов (на 20% дальше) для учёта текста
    # Но это может сильно расширить кадр, поэтому сделаем разумный запас
    # Просто увеличим лимиты на 20% после расчёта.

    all_points_arr = np.array(all_points)
    min_vals = np.min(all_points_arr, axis=0)
    max_vals = np.max(all_points_arr, axis=0)
    max_range = np.max(max_vals - min_vals)

    # Длина осей – 20% от размаха (чтобы не занимали много места)
    axis_length = max_range * 0.2

    # Добавляем концы осей в список точек
    axis_ends = [[axis_length,0,0], [0,axis_length,0], [0,0,axis_length]]
    all_points_arr = np.vstack([all_points_arr, axis_ends])
    min_vals = np.min(all_points_arr, axis=0)
    max_vals = np.max(all_points_arr, axis=0)
    # Увеличиваем лимиты на 20% для учёта текста
    span = max_vals - min_vals
    margin = 0.2 * span
    ax.set_xlim([min_vals[0]-margin[0], max_vals[0]+margin[0]])
    ax.set_ylim([min_vals[1]-margin[1], max_vals[1]+margin[1]])
    ax.set_zlim([min_vals[2]-margin[2], max_vals[2]+margin[2]])

    # Рисуем оси после установки лимитов
    draw_axes(ax, origin=(0,0,0), length=axis_length, color='black')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join('static', filename), dpi=150)
    plt.close()

def generate_all_plots(data):
    """Создаёт три графика для отчёта."""
    # 1. Траектория (увеличиваем временной интервал до 2.5)
    t_vals = np.linspace(0, 2.5, 200)
    x_t = 8 * np.cos(np.pi * t_vals**2 / 3)
    y_t = 16 * np.sin(np.pi * t_vals**2 / 3)
    z_t = 2 * t_vals**2 + 4

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_t, y_t, z_t, label='Траектория точки')
    ax.scatter(data['point'][0], data['point'][1], data['point'][2],
               color='red', s=50, label='Положение при t=1')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Способ задания движения (абсолютная траектория)')

    # Собираем точки траектории, точку M и начало координат для расчёта лимитов
    all_points = np.vstack([np.column_stack([x_t, y_t, z_t]),
                            [data['point']],
                            [0,0,0]])
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    max_range = np.max(max_vals - min_vals)
    axis_length = max_range * 0.2

    # Добавляем концы осей и вектор V_abs
    V_abs = data['V_abs']
    end_V_abs = np.array(data['point']) + V_abs
    axis_ends = [[axis_length,0,0], [0,axis_length,0], [0,0,axis_length], end_V_abs]
    all_points = np.vstack([all_points, axis_ends])
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    span = max_vals - min_vals
    margin = 0.2 * span
    ax.set_xlim([min_vals[0]-margin[0], max_vals[0]+margin[0]])
    ax.set_ylim([min_vals[1]-margin[1], max_vals[1]+margin[1]])
    ax.set_zlim([min_vals[2]-margin[2], max_vals[2]+margin[2]])

    draw_axes(ax, origin=(0,0,0), length=axis_length, color='black')
    ax.grid(True, alpha=0.3)

    # Вектор абсолютной скорости
    ax.quiver(data['point'][0], data['point'][1], data['point'][2],
              V_abs[0], V_abs[1], V_abs[2],
              color='purple', label='V_abs', arrow_length_ratio=0.03)

    # Подпись для V_abs с перпендикулярным смещением
    norm_vec = V_abs / (np.linalg.norm(V_abs) + 1e-8)
    perp = np.array([-norm_vec[1], norm_vec[0], 0])
    if np.linalg.norm(perp) < 0.1:
        perp = np.array([1, 0, 0])
    perp = perp / np.linalg.norm(perp) * 0.2
    text_pos = end_V_abs + perp
    ax.text(text_pos[0], text_pos[1], text_pos[2],
            f'V_abs = {data["V_abs_mod"]:.2f}',
            color='purple', fontsize=8, ha='center', va='center')

    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'trajectory.png'), dpi=150)
    plt.close()

    # 2. Скорости
    point = data['point']
    vectors = [data['V_rel'], data['V_rot'], data['V_trans_post'], data['V_abs']]
    colors = ['blue', 'green', 'orange', 'purple']
    labels = ['V_rel', 'V_rot', 'V_trans_post', 'V_abs']
    numeric_vals = [data['V_rel_mod'], data['V_rot_mod'], data['V_trans_post_mod'], data['V_abs_mod']]
    plot_vectors('Векторы скоростей в точке M', point, vectors, colors, labels,
                 'velocities.png', numeric_values=numeric_vals)

    # 3. Ускорения
    vectors = [data['a_rel'], data['a_centr'], data['a_rot'],
               data['a_trans_post'], data['a_cor'], data['a_abs']]
    colors = ['blue', 'green', 'orange', 'brown', 'cyan', 'purple']
    labels = ['a_rel', 'a_centr', 'a_rot', 'a_trans_post', 'a_cor', 'a_abs']
    numeric_vals = [data['a_rel_mod'], data['a_centr_mod'], data['a_rot_mod'],
                    data['a_trans_post_mod'], data['a_cor_mod'], data['a_abs_mod']]
    plot_vectors('Векторы ускорений в точке M', point, vectors, colors, labels,
                 'accelerations.png', numeric_values=numeric_vals)
