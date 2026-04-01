import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    """Класс для рисования 3D-стрелок (для дуги вращения)."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def draw_axes(ax, origin=(0,0,0), length=None, color='black', labels=['X', 'Y', 'Z']):
    """Рисует оси координат заданной длины (length)."""
    for i, label in enumerate(labels):
        vec = [0,0,0]
        vec[i] = length
        ax.quiver(origin[0], origin[1], origin[2],
                  vec[0], vec[1], vec[2],
                  color=color, label=label, arrow_length_ratio=0.03, linewidth=2)
        ax.text(origin[0]+vec[0], origin[1]+vec[1], origin[2]+vec[2], label,
                color=color, fontsize=10, ha='center', va='center')

def plot_vectors(title, point, vectors, colors, labels, filename, numeric_values=None):
    """Остаётся без изменений (см. предыдущую версию)."""
    # ... (код из предыдущего ответа) ...

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
    ax.set_xlabel("X'")
    ax.set_ylabel("Y'")
    ax.set_zlabel("Z'")
    ax.set_title('Способ задания движения (абсолютная траектория)')

    # Точка O – начало подвижной системы координат
    O = np.array([0.0, 0.0, data['zp']])   # zp = z'(t) при t=1

    # Собираем все точки для расчёта лимитов (траектория, M, O, концы осей)
    all_points = np.vstack([np.column_stack([x_t, y_t, z_t]),
                            [data['point']],
                            [0,0,0],    # начало неподвижной системы
                            O])          # начало подвижной системы

    # Предварительный размах для длины осей
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    max_range = np.max(max_vals - min_vals)
    axis_length = max_range * 0.25

    # Концы чёрных осей (неподвижной системы)
    axis_ends = [[axis_length,0,0], [0,axis_length,0], [0,0,axis_length]]
    all_points = np.vstack([all_points, axis_ends])

    # Концы красных осей (подвижной системы) – они исходят из O
    red_axis_ends = [O + np.array([axis_length,0,0]),
                     O + np.array([0,axis_length,0]),
                     O + np.array([0,0,axis_length])]
    all_points = np.vstack([all_points, red_axis_ends])

    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    margin = 0.2
    ax.set_xlim([min_vals[0]-margin, max_vals[0]+margin])
    ax.set_ylim([min_vals[1]-margin, max_vals[1]+margin])
    ax.set_zlim([min_vals[2]-margin, max_vals[2]+margin])

    # Чёрные оси неподвижной системы
    draw_axes(ax, origin=(0,0,0), length=axis_length, color='black',
              labels=["X'", "Y'", "Z'"])

    # Красные оси подвижной системы из точки O
    draw_axes(ax, origin=O, length=axis_length, color='red',
              labels=['X', 'Y', 'Z'])

    # Точка O (подвижное начало)
    ax.scatter(O[0], O[1], O[2], color='blue', s=40, label='O (начало подвижной)')
    ax.text(O[0], O[1], O[2], ' O', color='blue', fontsize=10)

    # Вектор абсолютной скорости (нормированный, короткий)
    V_abs = data['V_abs']
    V_norm = V_abs / (np.linalg.norm(V_abs) + 1e-8)
    arrow_length = axis_length * 0.4
    V_scaled = V_norm * arrow_length
    ax.quiver(data['point'][0], data['point'][1], data['point'][2],
              V_scaled[0], V_scaled[1], V_scaled[2],
              color='purple', label='V_abs', arrow_length_ratio=0.03)
    end = np.array(data['point']) + V_scaled
    offset = V_norm * 0.1
    text_pos = end + offset
    ax.text(text_pos[0], text_pos[1], text_pos[2],
            f'V_abs = {data["V_abs_mod"]:.2f}',
            color='purple', fontsize=8, ha='center', va='center')

    # Направление вращения подвижной системы (дуга со стрелкой)
    # Угловая скорость ω = -2π (по часовой стрелке, если смотреть сверху)
    # Рисуем дугу в плоскости X'Y' вокруг точки O, на высоте Z = O[2] + небольшое смещение
    theta = np.linspace(0, -np.pi/2, 50)  # четверть окружности по часовой стрелке
    radius = axis_length * 0.6
    x_arc = O[0] + radius * np.cos(theta)
    y_arc = O[1] + radius * np.sin(theta)
    z_arc = O[2] + axis_length * 0.05  # чуть выше, чтобы не накладывалось на оси
    ax.plot(x_arc, y_arc, z_arc, color='green', linewidth=2)
    # Добавляем стрелку в конце дуги
    arrow = Arrow3D([x_arc[-2], x_arc[-1]], [y_arc[-2], y_arc[-1]],
                    [z_arc[-2], z_arc[-1]],
                    mutation_scale=20, lw=2, arrowstyle='->', color='green')
    ax.add_artist(arrow)
    # Подпись ω
    mid = len(theta)//2
    ax.text(x_arc[mid], y_arc[mid], z_arc[mid]+0.02, f'ω = {data["omega"]:.2f}',
            color='green', fontsize=9)

    ax.grid(True, alpha=0.3)
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
