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
    # 1. Траектория (неподвижная система)
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

    # Дуга вращения
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
    # 2. Скорости (подвижная система, без осей X,Y,Z,
    #    базисные векторы ω, a_кор, V_отн)
    # ------------------------------------------------------------
    point = data['point']
    vectors = [data['V_rel'], data['V_rot'], data['V_trans_post'], data['V_abs']]
    colors = ['blue', 'green', 'orange', 'purple']
    labels = ['V_rel', 'V_rot', 'V_trans_post', 'V_abs']
    numeric_vals = [data['V_rel_mod'], data['V_rot_mod'],
                    data['V_trans_post_mod'], data['V_abs_mod']]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Векторы скоростей в точке M')

    # Точка M
    ax.scatter(point[0], point[1], point[2], color='red', s=50, label='Point M')

    # Все векторы скоростей
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

    # Базисные векторы (ω, a_кор, V_отн) из точки M
    omega_vec = np.array([0.0, 0.0, data['omega']])
    a_cor_vec = data['a_cor']
    V_rel_vec = data['V_rel']

    # Вычисляем базовую длину для масштабирования базисных векторов
    # на основе размаха реальных векторов скоростей (без учёта исходных огромных значений)
    all_points_for_scale = [np.array(point)] + [np.array(point) + v for v in vectors] + [np.array([0,0,0])]
    all_arr = np.array(all_points_for_scale)
    max_range = np.max(np.max(all_arr, axis=0) - np.min(all_arr, axis=0))
    basis_length = max_range * 0.25

    def scaled_basis(vec, target_length):
        if np.linalg.norm(vec) < 1e-8:
            return np.zeros(3)
        return vec / np.linalg.norm(vec) * target_length

    # ω делаем заметно короче, чтобы он не выходил за пределы
    omega_basis = scaled_basis(omega_vec, basis_length * 0.3)
    a_cor_basis = scaled_basis(a_cor_vec, basis_length)
    V_rel_basis = scaled_basis(V_rel_vec, basis_length)

    # Рисуем базисные векторы
    ax.quiver(point[0], point[1], point[2],
              omega_basis[0], omega_basis[1], omega_basis[2],
              color='magenta', label='ω (basis)', arrow_length_ratio=0.05)
    end_omega = np.array(point) + omega_basis
    ax.text(end_omega[0], end_omega[1], end_omega[2],
            f'ω = {data["omega"]:.2f}', color='magenta', fontsize=8)

    ax.quiver(point[0], point[1], point[2],
              a_cor_basis[0], a_cor_basis[1], a_cor_basis[2],
              color='cyan', label='a_кор (basis)', arrow_length_ratio=0.05)
    end_acor = np.array(point) + a_cor_basis
    ax.text(end_acor[0], end_acor[1], end_acor[2],
            f'a_кор = {data["a_cor_mod"]:.2f}', color='cyan', fontsize=8)

    ax.quiver(point[0], point[1], point[2],
              V_rel_basis[0], V_rel_basis[1], V_rel_basis[2],
              color='lime', label='V_отн (basis)', arrow_length_ratio=0.05)
    end_vrel = np.array(point) + V_rel_basis
    ax.text(end_vrel[0], end_vrel[1], end_vrel[2],
            f'V_отн = {data["V_rel_mod"]:.2f}', color='lime', fontsize=8)

    # Устанавливаем лимиты с учётом всех нарисованных объектов:
    # - концы реальных векторов скоростей (vectors)
    # - концы масштабированных базисных векторов
    # - точка M и начало координат
    all_points = [np.array(point)]
    for v in vectors:
        all_points.append(np.array(point) + v)
    all_points.append(np.array(point) + omega_basis)
    all_points.append(np.array(point) + a_cor_basis)
    all_points.append(np.array(point) + V_rel_basis)
    all_points.append(np.array([0,0,0]))
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
    # 3. Ускорения (подвижная система, с осями X,Y,Z)
    # ------------------------------------------------------------
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

    # Добавляем вектор ω из точки O
    O = np.array([0.0, 0.0, data['zp']])
    omega_vec = np.array([0.0, 0.0, data['omega']])
    # Масштаб для ω на третьем рисунке (можно оставить прежним)
    all_points_for_scale = [np.array(point)] + [np.array(point) + v for v in vectors] + [O, O + omega_vec, np.array([0,0,0])]
    all_arr = np.array(all_points_for_scale)
    max_range = np.max(np.max(all_arr, axis=0) - np.min(all_arr, axis=0))
    omega_length = max_range * 0.2
    omega_unit = omega_vec / (np.linalg.norm(omega_vec) + 1e-8)
    omega_scaled = omega_unit * omega_length
    ax.quiver(O[0], O[1], O[2],
              omega_scaled[0], omega_scaled[1], omega_scaled[2],
              color='magenta', label='ω', arrow_length_ratio=0.05)
    end_omega = O + omega_scaled
    ax.text(end_omega[0], end_omega[1], end_omega[2],
            f'ω = {data["omega"]:.2f}', color='magenta', fontsize=8)

    # Расчёт лимитов (включая все векторы ускорений, ω и оси)
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
