import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def draw_axes(ax, origin=(0,0,0), length=None, color='black', labels=['X', 'Y', 'Z']):
    for i, label in enumerate(labels):
        vec = [0,0,0]
        vec[i] = length
        ax.quiver(origin[0], origin[1], origin[2],
                  vec[0], vec[1], vec[2],
                  color=color, label=label, arrow_length_ratio=0.03, linewidth=2)
        ax.text(origin[0]+vec[0], origin[1]+vec[1], origin[2]+vec[2], label,
                color=color, fontsize=10, ha='center', va='center')

def generate_all_plots(data):
    # ------------------------------------------------------------
    # 1. Траектория (без изменений)
    # ------------------------------------------------------------
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

    O = np.array([0.0, 0.0, data['zp']])

    all_points = np.vstack([np.column_stack([x_t, y_t, z_t]),
                            [data['point']], [0,0,0], O])
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    max_range = np.max(max_vals - min_vals)
    axis_length = max_range * 0.25

    axis_ends = [[axis_length,0,0], [0,axis_length,0], [0,0,axis_length]]
    red_ends = [O + np.array([axis_length,0,0]),
                O + np.array([0,axis_length,0]),
                O + np.array([0,0,axis_length])]
    all_points = np.vstack([all_points, axis_ends, red_ends])
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    margin = 0.2
    ax.set_xlim([min_vals[0]-margin, max_vals[0]+margin])
    ax.set_ylim([min_vals[1]-margin, max_vals[1]+margin])
    ax.set_zlim([min_vals[2]-margin, max_vals[2]+margin])

    draw_axes(ax, origin=(0,0,0), length=axis_length, color='black',
              labels=["X'", "Y'", "Z'"])
    draw_axes(ax, origin=O, length=axis_length, color='red', labels=['X', 'Y', 'Z'])
    ax.scatter(O[0], O[1], O[2], color='blue', s=40, label='O (начало подвижной)')
    ax.text(O[0], O[1], O[2], ' O', color='blue', fontsize=10)

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

    theta = np.linspace(0, -np.pi/2, 50)
    radius = axis_length * 0.6
    x_arc = O[0] + radius * np.cos(theta)
    y_arc = O[1] + radius * np.sin(theta)
    z_arc = O[2] + axis_length * 0.05
    z_arc_arr = np.full_like(x_arc, z_arc)
    ax.plot(x_arc, y_arc, z_arc_arr, color='green', linewidth=2)
    arrow = Arrow3D([x_arc[-2], x_arc[-1]], [y_arc[-2], y_arc[-1]],
                    [z_arc_arr[-2], z_arc_arr[-1]],
                    mutation_scale=20, lw=2, arrowstyle='->', color='green')
    ax.add_artist(arrow)
    mid = len(theta)//2
    ax.text(x_arc[mid], y_arc[mid], z_arc_arr[mid]+0.02, f'ω = {data["omega"]:.2f}',
            color='green', fontsize=9)

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'trajectory.png'), dpi=150)
    plt.close()

    # ------------------------------------------------------------
    # 2. Скорости (новые оси: V_отн, a_cor, ω)
    # ------------------------------------------------------------
    point = data['point']
    V_rel = data['V_rel']
    V_rot = data['V_rot']
    V_trans_post = data['V_trans_post']
    V_abs = data['V_abs']

    # Базисные векторы для новой системы координат
    e_x = V_rel / np.linalg.norm(V_rel)
    e_y = data['a_cor'] / np.linalg.norm(data['a_cor'])
    e_z = np.array([0.0, 0.0, 1.0])

    # Приводим к ортонормированному базису (правому)
    e_z = e_z / np.linalg.norm(e_z)
    e_x = e_x / np.linalg.norm(e_x)
    e_y = np.cross(e_z, e_x)
    e_y = e_y / np.linalg.norm(e_y)

    R = np.array([e_x, e_y, e_z])

    def transform(v):
        return R @ v

    point_new = transform(point)
    V_rel_new = transform(V_rel)
    V_rot_new = transform(V_rot)
    V_trans_post_new = transform(V_trans_post)
    V_abs_new = transform(V_abs)

    vectors_new = [V_rel_new, V_rot_new, V_trans_post_new, V_abs_new]
    colors = ['blue', 'green', 'orange', 'purple']
    labels = ['V_rel', 'V_rot', 'V_trans_post', 'V_abs']
    numeric_vals = [data['V_rel_mod'], data['V_rot_mod'],
                    data['V_trans_post_mod'], data['V_abs_mod']]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('V_отн')
    ax.set_ylabel('a_cor')
    ax.set_zlabel('ω')
    ax.set_title('Векторы скоростей в точке M (оси: V_отн, a_cor, ω)')

    ax.scatter(point_new[0], point_new[1], point_new[2], color='red', s=50, label='Point M')

    for idx, (vec, color, label) in enumerate(zip(vectors_new, colors, labels)):
        ax.quiver(point_new[0], point_new[1], point_new[2],
                  vec[0], vec[1], vec[2],
                  color=color, label=label, arrow_length_ratio=0.03)
        if numeric_vals and idx < len(numeric_vals):
            end = np.array(point_new) + vec
            offset = vec / (np.linalg.norm(vec) + 1e-8) * 0.1
            text_pos = end + offset
            ax.text(text_pos[0], text_pos[1], text_pos[2],
                    f'{label} = {numeric_vals[idx]:.2f}',
                    color=color, fontsize=8, ha='center', va='center')

    # Определяем масштаб для отрисовки осей
    all_points = [point_new] + [point_new + v for v in vectors_new] + [np.zeros(3)]
    all_arr = np.array(all_points)
    min_vals = np.min(all_arr, axis=0)
    max_vals = np.max(all_arr, axis=0)
    max_range = np.max(max_vals - min_vals)
    axis_length_new = max_range * 0.25

    # Рисуем оси из точки M
    ax.quiver(point_new[0], point_new[1], point_new[2],
              axis_length_new, 0, 0,
              color='black', arrow_length_ratio=0.05, linewidth=2)
    ax.text(point_new[0] + axis_length_new, point_new[1], point_new[2],
            'V_отн', color='black', fontsize=10, ha='center', va='center')
    ax.quiver(point_new[0], point_new[1], point_new[2],
              0, axis_length_new, 0,
              color='black', arrow_length_ratio=0.05, linewidth=2)
    ax.text(point_new[0], point_new[1] + axis_length_new, point_new[2],
            'a_cor', color='black', fontsize=10, ha='center', va='center')
    ax.quiver(point_new[0], point_new[1], point_new[2],
              0, 0, axis_length_new,
              color='black', arrow_length_ratio=0.05, linewidth=2)
    ax.text(point_new[0], point_new[1], point_new[2] + axis_length_new,
            'ω', color='black', fontsize=10, ha='center', va='center')

    all_points = [point_new] + [point_new + v for v in vectors_new] + \
                 [point_new + np.array([axis_length_new,0,0]),
                  point_new + np.array([0,axis_length_new,0]),
                  point_new + np.array([0,0,axis_length_new]),
                  np.zeros(3)]
    all_arr = np.array(all_points)
    min_vals = np.min(all_arr, axis=0)
    max_vals = np.max(all_arr, axis=0)
    margin = 0.2
    ax.set_xlim([min_vals[0]-margin, max_vals[0]+margin])
    ax.set_ylim([min_vals[1]-margin, max_vals[1]+margin])
    ax.set_zlim([min_vals[2]-margin, max_vals[2]+margin])

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'velocities.png'), dpi=150)
    plt.close()

    # ------------------------------------------------------------
    # 3. Ускорения (новые оси: ω, a_cor, a_отн)
    # ------------------------------------------------------------
    # Векторы в исходной системе
    a_rel = data['a_rel']
    a_centr = data['a_centr']
    a_rot = data['a_rot']
    a_trans_post = data['a_trans_post']
    a_cor = data['a_cor']
    a_abs = data['a_abs']

    # Базисные векторы для новой системы координат
    e_z = np.array([0.0, 0.0, 1.0])                      # ось ω (вертикаль)
    e_y = a_cor / np.linalg.norm(a_cor)                  # ось a_cor
    e_x = a_rel / np.linalg.norm(a_rel)                  # ось a_отн

    # Приводим к ортонормированному базису (правому)
    e_z = e_z / np.linalg.norm(e_z)
    e_x = e_x / np.linalg.norm(e_x)
    e_y = np.cross(e_z, e_x)
    e_y = e_y / np.linalg.norm(e_y)

    R_acc = np.array([e_x, e_y, e_z])

    def transform_acc(v):
        return R_acc @ v

    point_acc_new = transform_acc(point)
    a_rel_new = transform_acc(a_rel)
    a_centr_new = transform_acc(a_centr)
    a_rot_new = transform_acc(a_rot)
    a_trans_post_new = transform_acc(a_trans_post)
    a_cor_new = transform_acc(a_cor)
    a_abs_new = transform_acc(a_abs)

    vectors_acc_new = [a_rel_new, a_centr_new, a_rot_new,
                       a_trans_post_new, a_cor_new, a_abs_new]
    colors_acc = ['blue', 'green', 'orange', 'brown', 'cyan', 'purple']
    labels_acc = ['a_rel', 'a_centr', 'a_rot', 'a_trans_post', 'a_cor', 'a_abs']
    numeric_vals_acc = [data['a_rel_mod'], data['a_centr_mod'], data['a_rot_mod'],
                        data['a_trans_post_mod'], data['a_cor_mod'], data['a_abs_mod']]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('a_отн')
    ax.set_ylabel('a_cor')
    ax.set_zlabel('ω')
    ax.set_title('Векторы ускорений в точке M (оси: a_отн, a_cor, ω)')

    ax.scatter(point_acc_new[0], point_acc_new[1], point_acc_new[2], color='red', s=50, label='Point M')

    for idx, (vec, color, label) in enumerate(zip(vectors_acc_new, colors_acc, labels_acc)):
        ax.quiver(point_acc_new[0], point_acc_new[1], point_acc_new[2],
                  vec[0], vec[1], vec[2],
                  color=color, label=label, arrow_length_ratio=0.03)
        if numeric_vals_acc and idx < len(numeric_vals_acc):
            end = np.array(point_acc_new) + vec
            offset = vec / (np.linalg.norm(vec) + 1e-8) * 0.1
            text_pos = end + offset
            ax.text(text_pos[0], text_pos[1], text_pos[2],
                    f'{label} = {numeric_vals_acc[idx]:.2f}',
                    color=color, fontsize=8, ha='center', va='center')

    # Определяем масштаб для осей
    all_points = [point_acc_new] + [point_acc_new + v for v in vectors_acc_new] + [np.zeros(3)]
    all_arr = np.array(all_points)
    min_vals = np.min(all_arr, axis=0)
    max_vals = np.max(all_arr, axis=0)
    max_range = np.max(max_vals - min_vals)
    axis_length_new = max_range * 0.25

    # Рисуем оси из точки M
    ax.quiver(point_acc_new[0], point_acc_new[1], point_acc_new[2],
              axis_length_new, 0, 0,
              color='black', arrow_length_ratio=0.05, linewidth=2)
    ax.text(point_acc_new[0] + axis_length_new, point_acc_new[1], point_acc_new[2],
            'a_отн', color='black', fontsize=10, ha='center', va='center')
    ax.quiver(point_acc_new[0], point_acc_new[1], point_acc_new[2],
              0, axis_length_new, 0,
              color='black', arrow_length_ratio=0.05, linewidth=2)
    ax.text(point_acc_new[0], point_acc_new[1] + axis_length_new, point_acc_new[2],
            'a_cor', color='black', fontsize=10, ha='center', va='center')
    ax.quiver(point_acc_new[0], point_acc_new[1], point_acc_new[2],
              0, 0, axis_length_new,
              color='black', arrow_length_ratio=0.05, linewidth=2)
    ax.text(point_acc_new[0], point_acc_new[1], point_acc_new[2] + axis_length_new,
            'ω', color='black', fontsize=10, ha='center', va='center')

    # Устанавливаем лимиты
    all_points = [point_acc_new] + [point_acc_new + v for v in vectors_acc_new] + \
                 [point_acc_new + np.array([axis_length_new,0,0]),
                  point_acc_new + np.array([0,axis_length_new,0]),
                  point_acc_new + np.array([0,0,axis_length_new]),
                  np.zeros(3)]
    all_arr = np.array(all_points)
    min_vals = np.min(all_arr, axis=0)
    max_vals = np.max(all_arr, axis=0)
    margin = 0.2
    ax.set_xlim([min_vals[0]-margin, max_vals[0]+margin])
    ax.set_ylim([min_vals[1]-margin, max_vals[1]+margin])
    ax.set_zlim([min_vals[2]-margin, max_vals[2]+margin])

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'accelerations.png'), dpi=150)
    plt.close()
