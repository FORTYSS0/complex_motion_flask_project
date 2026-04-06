import numpy as np
import sympy as sp


# Символьные переменные и выражения
t_sym = sp.Symbol('t', real=True)
a_sym, b_sym = sp.symbols('a b', real=True)

# Параметры движения
a_val = 8
b_val = -2

# Относительное движение
x_sym = a_sym * sp.cos(b_sym * sp.pi * t_sym**2 / 6)
y_sym = -2 * a_sym * sp.sin(b_sym * sp.pi * t_sym**2 / 6)

x_expr = x_sym.subs({a_sym: a_val, b_sym: b_val}).simplify()
y_expr = y_sym.subs({a_sym: a_val, b_sym: b_val}).simplify()

vx_expr = sp.diff(x_expr, t_sym).simplify()
vy_expr = sp.diff(y_expr, t_sym).simplify()
ax_expr = sp.diff(vx_expr, t_sym).simplify()
ay_expr = sp.diff(vy_expr, t_sym).simplify()

# Переносное движение
z_expr = 2 * t_sym**2 + 4
vz_expr = sp.diff(z_expr, t_sym)
az_expr = sp.diff(vz_expr, t_sym)

phi_expr = 2 * sp.sin(sp.pi * t_sym)
omega_expr = sp.diff(phi_expr, t_sym).simplify()
alpha_expr = sp.diff(omega_expr, t_sym).simplify()


# Вычисление в конкретный момент времени
def compute_values_at_time(t0=1):
    """Вычисляет все кинематические величины в момент времени t0."""
    x0 = float(x_expr.subs(t_sym, t0))
    y0 = float(y_expr.subs(t_sym, t0))
    z0 = 0.0
    
    vx0 = float(vx_expr.subs(t_sym, t0))
    vy0 = float(vy_expr.subs(t_sym, t0))
    V_rel = np.array([vx0, vy0, 0.0])
    
    ax0 = float(ax_expr.subs(t_sym, t0))
    ay0 = float(ay_expr.subs(t_sym, t0))
    a_rel = np.array([ax0, ay0, 0.0])
    
    zp0 = float(z_expr.subs(t_sym, t0))
    vz0 = float(vz_expr.subs(t_sym, t0))
    az0 = float(az_expr.subs(t_sym, t0))
    
    phi0 = float(phi_expr.subs(t_sym, t0))
    omega0 = float(omega_expr.subs(t_sym, t0))
    alpha0 = float(alpha_expr.subs(t_sym, t0))
    
    return {
        'x0': x0, 'y0': y0, 'z0': z0,
        'vx0': vx0, 'vy0': vy0,
        'ax0': ax0, 'ay0': ay0,
        'zp0': zp0, 'vz0': vz0, 'az0': az0,
        'phi0': phi0, 'omega0': omega0, 'alpha0': alpha0
    }


# Основная функция расчёта сложного движения
def psk_compute_complex_motion(t=1):
    """Вычисляет все кинематические характеристики сложного движения точки."""
    vals = compute_values_at_time(t)
    
    # Векторы относительного движения
    r_rel = np.array([vals['x0'], vals['y0'], 0.0])
    V_rel = np.array([vals['vx0'], vals['vy0'], 0.0])
    a_rel = np.array([vals['ax0'], vals['ay0'], 0.0])
    
    # Переносное движение (поступательное)
    V_trans_post = np.array([0.0, 0.0, vals['vz0']])
    a_trans_post = np.array([0.0, 0.0, vals['az0']])
    
    # Переносное движение (вращательное)
    V_rot = np.cross([0, 0, vals['omega0']], r_rel)
    a_centr = np.cross([0, 0, vals['omega0']], np.cross([0, 0, vals['omega0']], r_rel))
    a_rot = np.cross([0, 0, vals['alpha0']], r_rel)
    
    # Абсолютное движение
    V_abs = V_rel + V_trans_post + V_rot
    a_trans = a_centr + a_rot + a_trans_post
    a_cor = 2 * np.cross([0, 0, vals['omega0']], V_rel)
    a_abs = a_rel + a_trans + a_cor
    
    # Радиус кривизны
    V_abs_norm = np.linalg.norm(V_abs)
    cross = np.cross(V_abs, a_abs)
    cross_norm = np.linalg.norm(cross)
    rho = V_abs_norm**3 / cross_norm if cross_norm != 0 else float('inf')
    
    # Сбор результатов
    data = {
        't': t,
        'point': np.array([vals['x0'], vals['y0'], vals['zp0']]),
        'zp': vals['zp0'],
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
        'omega': vals['omega0'],
        'alpha': vals['alpha0'],
        'dzpdt': vals['vz0'],
        'd2zpdt2': vals['az0'],
        'phi': vals['phi0'],
        'dphi_dt': vals['omega0'],
        'd2phi_dt2': vals['alpha0'],
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
    
    # Формулы для отображения
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
    
    return data, formulas
