import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_axes(ax, origin=(0,0,0), length_scale=0.3, color='black', labels=['X', 'Y', 'Z']):
    """
    Рисует оси координат из точки origin.
    Длина подстраивается под текущие лимиты осей, масштабируется length_scale.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    max_range = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0])
    length = max_range * length_scale
    for i, label in enumerate(labels):
        vec = [0,0,0]
        vec[i] = length
        ax.quiver(origin[0], origin[1], origin[2],
                  vec[0], vec[1], vec[2],
                  color=color, label=label,
                  arrow_length_ratio=0.08, linewidth=2)
        ax.text(origin[0]+vec[0], origin[1]+vec[1], origin[2]+vec[2], label,
                color=color, fontsize=12, ha='center', va='center')

def plot_vectors(title, point, vectors, colors, labels, norms, filename):
    """Сохраняет 3D-график с векторами из точки. norms — список модулей для подписей."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Точка M
    ax.scatter(point[0], point[1], point[2], color='red', s=50, label='Point M')

    # Векторы с подписями, включающими численные значения
    for vec, color, base_label, norm in zip(vectors, colors, labels, norms):
        label = f'{base_label} = {norm:.2f} м/с' if 'V' in base_label else f'{base_label} = {norm:.2f} м/с²'
        ax.quiver(point[0], point[1], point[2],
                  vec[0], vec[1], vec[2],
                  color=color, label=label, arrow_length_ratio=0.08)

    # Автоматическое масштабирование
    all_points = [point] + [point + v for v in vectors]
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    margin = 0.2
    ax.set_xlim([min_vals[0]-margin, max_vals[0]+margin])
    ax.set_ylim([min_vals[1]-margin, max_vals[1]+margin])
    ax.set_zlim([min_vals[2]-margin, max_vals[2]+margin])

    # Оси рисуем после установки лимитов, с уменьшенным масштабом (0.3)
    draw_axes(ax, origin=(0,0,0), length_scale=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend()

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

    # Добавляем вектор абсолютной скорости из точки M
    point = data['point']
    V_abs = data['V_abs']
    V_abs_mod = data['V_abs_mod']
    ax.quiver(point[0], point[1], point[2],
              V_abs[0], V_abs[1], V_abs[2],
              color='purple', label=f'V_abs = {V_abs_mod:.2f} м/с',
              arrow_length_ratio=0.08)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Способ задания движения (абсолютная траектория)')

    # Масштабируем по данным траектории
    all_points = np.vstack([np.column_stack([x_t, y_t, z_t]), [point], [point + V_abs]])
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    margin = 0.2
    ax.set_xlim([min_vals[0]-margin, max_vals[0]+margin])
    ax.set_ylim([min_vals[1]-margin, max_vals[1]+margin])
    ax.set_zlim([min_vals[2]-margin, max_vals[2]+margin])

    draw_axes(ax, origin=(0,0,0), length_scale=0.4)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'trajectory.png'), dpi=150)
    plt.close()

    # 2. Скорости
    point = data['point']
    vectors = [data['V_rel'], data['V_rot'], data['V_trans_post'], data['V_abs']]
    colors = ['blue', 'green', 'orange', 'purple']
    labels = ['V_rel', 'V_rot', 'V_trans_post', 'V_abs']
    norms = [data['V_rel_mod'], data['V_rot_mod'], data['V_trans_post_mod'], data['V_abs_mod']]
    plot_vectors('Векторы скоростей в точке M', point, vectors, colors, labels, norms,
                 'velocities.png')

    # 3. Ускорения
    vectors = [data['a_rel'], data['a_centr'], data['a_rot'],
               data['a_trans_post'], data['a_cor'], data['a_abs']]
    colors = ['blue', 'green', 'orange', 'brown', 'cyan', 'purple']
    labels = ['a_rel', 'a_centr', 'a_rot', 'a_trans_post', 'a_cor', 'a_abs']
    norms = [data['a_rel_mod'], data['a_centr_mod'], data['a_rot_mod'],
             data['a_trans_post_mod'], data['a_cor_mod'], data['a_abs_mod']]
    plot_vectors('Векторы ускорений в точке M', point, vectors, colors, labels, norms,
                 'accelerations.png')
