import numpy as np
import sympy as sp

def compute_relative_motion(t=1):
    """Возвращает положение, скорость, ускорение точки в подвижной системе."""
    # Относительное движение (из условия)
    x = 8 * np.cos(np.pi * t**2 / 3)
    y = 16 * np.sin(np.pi * t**2 / 3)
    z = 0

    # Символьное дифференцирование
    ts = sp.Symbol('t')
    x_sym = 8 * sp.cos(sp.pi * ts**2 / 3)
    y_sym = 16 * sp.sin(sp.pi * ts**2 / 3)
    dxdt_sym = sp.diff(x_sym, ts)
    dydt_sym = sp.diff(y_sym, ts)
    dxdt = float(dxdt_sym.subs(ts, t))
    dydt = float(dydt_sym.subs(ts, t))

    d2xdt2_sym = sp.diff(dxdt_sym, ts)
    d2ydt2_sym = sp.diff(dydt_sym, ts)
    d2xdt2 = float(d2xdt2_sym.subs(ts, t))
    d2ydt2 = float(d2ydt2_sym.subs(ts, t))

    V_rel = np.array([dxdt, dydt, 0.0])
    a_rel = np.array([d2xdt2, d2ydt2, 0.0])
    point_rel = (x, y, z)

    # Формулы для LaTeX
    formulas = {
        'x': sp.latex(x_sym),
        'y': sp.latex(y_sym),
        'dxdt': sp.latex(dxdt_sym),
        'dydt': sp.latex(dydt_sym),
        'd2xdt2': sp.latex(d2xdt2_sym),
        'd2ydt2': sp.latex(d2ydt2_sym),
    }
    return point_rel, V_rel, a_rel, formulas

def compute_portable_motion(t=1):
    """Возвращает параметры переносного движения."""
    zp = 2*t**2 + 4
    dzpdt = 4*t
    d2zpdt2 = 4
    phi = 2 * np.sin(np.pi * t)
    dphi_dt = 2 * np.pi * np.cos(np.pi * t)
    d2phi_dt2 = -2 * np.pi**2 * np.sin(np.pi * t)

    omega = dphi_dt
    alpha = d2phi_dt2

    # Символьные выражения для формул
    ts = sp.Symbol('t')
    zp_sym = 2*ts**2 + 4
    phi_sym = 2 * sp.sin(sp.pi * ts)
    formulas = {
        'zp': sp.latex(zp_sym),
        'dzpdt': sp.latex(sp.diff(zp_sym, ts)),
        'd2zpdt2': sp.latex(sp.diff(sp.diff(zp_sym, ts), ts)),
        'phi': sp.latex(phi_sym),
        'dphi_dt': sp.latex(sp.diff(phi_sym, ts)),
        'd2phi_dt2': sp.latex(sp.diff(sp.diff(phi_sym, ts), ts)),
    }
    return zp, dzpdt, d2zpdt2, phi, omega, alpha, formulas

def compute_complex_motion(t=1):
    """Основная функция, возвращающая все данные для отчёта."""
    point_rel, V_rel, a_rel, rel_formulas = compute_relative_motion(t)
    zp, dzpdt, d2zpdt2, phi, omega, alpha, port_formulas = compute_portable_motion(t)

    x, y, z_rel = point_rel
    r_rel = np.array([x, y, 0.0])

    # Переносная скорость
    V_trans_post = np.array([0.0, 0.0, dzpdt])
    V_rot = np.cross([0, 0, omega], r_rel)
    V_abs = V_rel + V_trans_post + V_rot

    # Переносное ускорение
    a_centr = np.cross([0, 0, omega], np.cross([0, 0, omega], r_rel))
    a_rot = np.cross([0, 0, alpha], r_rel)
    a_trans_post = np.array([0.0, 0.0, d2zpdt2])
    a_trans = a_centr + a_rot + a_trans_post

    # Кориолисово ускорение
    a_cor = 2 * np.cross([0, 0, omega], V_rel)

    # Абсолютное ускорение
    a_abs = a_rel + a_trans + a_cor

    # Радиус кривизны
    V_abs_norm = np.linalg.norm(V_abs)
    cross = np.cross(V_abs, a_abs)
    cross_norm = np.linalg.norm(cross)
    rho = V_abs_norm**3 / cross_norm if cross_norm != 0 else float('inf')

    # Положение точки в неподвижной системе
    point_abs = (x, y, zp)

    # Собираем все данные в словарь
    data = {
        't': t,
        'point': point_abs,
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
        'omega': omega,
        'alpha': alpha,
        'dzpdt': dzpdt,
        'd2zpdt2': d2zpdt2,
        'phi': phi,
        'dphi_dt': omega,
        'd2phi_dt2': alpha,
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
    # Объединяем формулы
    formulas = {**rel_formulas, **port_formulas}
    return data, formulas
