import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_axes(ax, origin=(0,0,0), length=None, color='black', labels=['X', 'Y', 'Z']):
    """Рисует оси координат из точки origin с автоматическим подбором длины."""
    # Получаем текущие лимиты осей, чтобы определить длину
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    max_range = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0])
    if length is None:
        length = max_range * 0.2   # 20% от размаха
    for i, label in enumerate(labels):
        vec = [0,0,0]
        vec[i] = length
        ax.quiver(origin[0], origin[1], origin[2],
                  vec[0], vec[1], vec[2],
                  color=color, label=label, arrow_length_ratio=0.1, linewidth=2)
        # Подпись в конце оси
        ax.text(origin[0]+vec[0], origin[1]+vec[1], origin[2]+vec[2], label,
                color=color, fontsize=12, ha='center', va='center')

def plot_vectors(title, point, vectors, colors, labels, filename):
    """Сохраняет 3D-график с векторами из точки."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Отрисовка осей координат (базисных векторов) из начала координат
    draw_axes(ax, origin=(0,0,0), color='black')
    ax.grid(True, alpha=0.3)   # размерная сетка

    # Точка M
    ax.scatter(point[0], point[1], point[2], color='red', s=50, label='Point M')

    # Векторы
    for vec, color, label in zip(vectors, colors, labels):
        ax.quiver(point[0], point[1], point[2],
                  vec[0], vec[1], vec[2],
                  color=color, label=label, arrow_length_ratio=0.1)

    ax.legend()
    # Автоматическое масштабирование
    max_abs = max([abs(v) for vec in vectors for v in vec] + [abs(p) for p in point])
    lim = max_abs * 1.2
    ax.set_xlim([point[0]-lim, point[0]+lim])
    ax.set_ylim([point[1]-lim, point[1]+lim])
    ax.set_zlim([point[2]-lim, point[2]+lim])

    plt.tight_layout()
    plt.savefig(os.path.join('static', filename), dpi=150)
    plt.close()

def generate_all_plots(data):
    """Создаёт три графика для отчёта."""
    # 1. Траектория
    t_vals = np.linspace(0, 1.2, 100)
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
    draw_axes(ax, color='black')
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
    plot_vectors('Векторы скоростей в точке M', point, vectors, colors, labels,
                 'velocities.png')

    # 3. Ускорения
    vectors = [data['a_rel'], data['a_centr'], data['a_rot'],
               data['a_trans_post'], data['a_cor'], data['a_abs']]
    colors = ['blue', 'green', 'orange', 'brown', 'cyan', 'purple']
    labels = ['a_rel', 'a_centr', 'a_rot', 'a_trans_post', 'a_cor', 'a_abs']
    plot_vectors('Векторы ускорений в точке M', point, vectors, colors, labels,
                 'accelerations.png')
