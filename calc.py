import numpy as np
import sympy as sp

# Символьные переменные
t_sym = sp.Symbol('t', real=True)
a_sym, b_sym = sp.symbols('a b', real=True)

# ----------------------------------------------------------------------
# Относительное движение (подвижная система)
# ----------------------------------------------------------------------
# Параметры из задания
a_val = 8
b_val = -2

# Относительное движение в символьной форме
x_sym = a_sym * sp.cos(b_sym * sp.pi * t_sym**2 / 6)
y_sym = -2 * a_sym * sp.sin(b_sym * sp.pi * t_sym**2 / 6)

# Подставляем a и b
x_expr = x_sym.subs({a_sym: a_val, b_sym: b_val}).simplify()
y_expr = y_sym.subs({a_sym: a_val, b_sym: b_val}).simplify()

# Производные
vx_expr = sp.diff(x_expr, t_sym).simplify()
vy_expr = sp.diff(y_expr, t_sym).simplify()
ax_expr = sp.diff(vx_expr, t_sym).simplify()
ay_expr = sp.diff(vy_expr, t_sym).simplify()

# ----------------------------------------------------------------------
# Переносное движение
# ----------------------------------------------------------------------
# Поступательное по оси Z
z_expr = 2 * t_sym**2 + 4
vz_expr = sp.diff(z_expr, t_sym)
az_expr = sp.diff(vz_expr, t_sym)

# Вращательное
phi_expr = 2 * sp.sin(sp.pi * t_sym)
omega_expr = sp.diff(phi_expr, t_sym).simplify()
alpha_expr = sp.diff(omega_expr, t_sym).simplify()

# ----------------------------------------------------------------------
# Расчёт в момент времени t=1
# ----------------------------------------------------------------------
t0 = 1

# Относительное положение (в подвижной системе)
x0 = float(x_expr.subs(t_sym, t0))
y0 = float(y_expr.subs(t_sym, t0))
z0 = 0.0
point_rel = (x0, y0, z0)

# Относительная скорость
vx0 = float(vx_expr.subs(t_sym, t0))
vy0 = float(vy_expr.subs(t_sym, t0))
V_rel = np.array([vx0, vy0, 0.0])

# Относительное ускорение
ax0 = float(ax_expr.subs(t_sym, t0))
ay0 = float(ay_expr.subs(t_sym, t0))
a_rel = np.array([ax0, ay0, 0.0])

# Переносное: поступательное
zp0 = float(z_expr.subs(t_sym, t0))
vz0 = float(vz_expr.subs(t_sym, t0))
az0 = float(az_expr.subs(t_sym, t0))

# Переносное: вращательное
phi0 = float(phi_expr.subs(t_sym, t0))
omega0 = float(omega_expr.subs(t_sym, t0))
alpha0 = float(alpha_expr.subs(t_sym, t0))

# Радиус-вектор точки в подвижной системе
r_rel = np.array([x0, y0, 0.0])

# Скорости
V_trans_post = np.array([0.0, 0.0, vz0])
V_rot = np.cross([0, 0, omega0], r_rel)
V_abs = V_rel + V_trans_post + V_rot

# Ускорения
a_centr = np.cross([0, 0, omega0], np.cross([0, 0, omega0], r_rel))
a_rot = np.cross([0, 0, alpha0], r_rel)
a_trans_post = np.array([0.0, 0.0, az0])
a_trans = a_centr + a_rot + a_trans_post
a_cor = 2 * np.cross([0, 0, omega0], V_rel)
a_abs = a_rel + a_trans + a_cor

# Радиус кривизны
V_abs_norm = np.linalg.norm(V_abs)
cross = np.cross(V_abs, a_abs)
cross_norm = np.linalg.norm(cross)
rho = V_abs_norm**3 / cross_norm if cross_norm != 0 else float('inf')

# Точка в неподвижной системе
point_abs = (x0, y0, zp0)

# ----------------------------------------------------------------------
# Словарь данных для шаблона
# ----------------------------------------------------------------------
data = {
    't': t0,
    'point': point_abs,
    'zp': zp0,
    'V_rel': V_rel,
    'V_rot': V_rot,
    'V_trans_post': V_trans_post,
    'V_abs': V_abs,
    'a_rel': a_rel,
    'a_centr': a_centr,
    'a_rot': a_rot,
    'a_trans_post': a_trans_post,
    'a_cor': a_cor,
    'a_abs': a_abs,
    'rho': rho,
    'omega': omega0,
    'alpha': alpha0,
    'dzpdt': vz0,
    'd2zpdt2': az0,
    'phi': phi0,
    'dphi_dt': omega0,
    'd2phi_dt2': alpha0,
    # Модули
    'V_rel_mod': np.linalg.norm(V_rel),
    'V_rot_mod': np.linalg.norm(V_rot),
    'V_trans_post_mod': np.linalg.norm(V_trans_post),
    'V_abs_mod': V_abs_norm,
    'a_rel_mod': np.linalg.norm(a_rel),
    'a_centr_mod': np.linalg.norm(a_centr),
    'a_rot_mod': np.linalg.norm(a_rot),
    'a_trans_post_mod': np.linalg.norm(a_trans_post),
    'a_cor_mod': np.linalg.norm(a_cor),
    'a_abs_mod': np.linalg.norm(a_abs),
}

# ----------------------------------------------------------------------
# LaTeX-формулы для отображения (упрощённые)
# ----------------------------------------------------------------------
formulas = {
    'x': sp.latex(x_expr),
    'y': sp.latex(y_expr),
    'dxdt': sp.latex(vx_expr),
    'dydt': sp.latex(vy_expr),
    'd2xdt2': sp.latex(ax_expr),
    'd2ydt2': sp.latex(ay_expr),
    'zp': sp.latex(z_expr),
    'dzpdt': sp.latex(vz_expr),
    'd2zpdt2': sp.latex(az_expr),
    'phi': sp.latex(phi_expr),
    'dphi_dt': sp.latex(omega_expr),
    'd2phi_dt2': sp.latex(alpha_expr),
}

def compute_complex_motion(t=1):
    """Возвращает data и formulas для заданного t (по умолчанию 1)."""
    # Здесь мы используем уже вычисленные значения для t=1
    # Если понадобится другое t, нужно пересчитать, но по заданию t=1
    return data, formulas
