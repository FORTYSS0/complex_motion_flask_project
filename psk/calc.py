import numpy as np
import sympy as sp


# Символьные переменные и выражения
t_sym = sp.Symbol('t', real=True)
a_sym, b_sym = sp.symbols('a b', real=True)

# Параметры движения
a_val = 8
b_val = -2

# Движение точки в ДСК (исходные уравнения)
x_sym = a_sym * sp.cos(b_sym * sp.pi * t_sym**2 / 6)
y_sym = -2 * a_sym * sp.sin(b_sym * sp.pi * t_sym**2 / 6)

x_expr = x_sym.subs({a_sym: a_val, b_sym: b_val}).simplify()
y_expr = y_sym.subs({a_sym: a_val, b_sym: b_val}).simplify()

# Переход к полярным координатам
r_sym = sp.sqrt(x_expr**2 + y_expr**2).simplify()
phi_sym = sp.atan2(y_expr, x_expr)

# Первые производные (скорости в полярных координатах)
dr_dt_sym = sp.diff(r_sym, t_sym).simplify()
dphi_dt_sym = sp.diff(phi_sym, t_sym).simplify()

# Вторые производные (ускорения в полярных координатах)
d2r_dt2_sym = sp.diff(dr_dt_sym, t_sym).simplify()
d2phi_dt2_sym = sp.diff(dphi_dt_sym, t_sym).simplify()


def compute_values_at_time(t0=1):
    """Вычисляет все кинематические величины в момент времени t0 для ПСК."""
    # Координаты точки в ДСК (для отображения)
    x0 = float(x_expr.subs(t_sym, t0))
    y0 = float(y_expr.subs(t_sym, t0))
    
    # Полярные координаты
    r0 = float(r_sym.subs(t_sym, t0))
    phi0 = float(phi_sym.subs(t_sym, t0))
    
    # Скорости
    dr_dt0 = float(dr_dt_sym.subs(t_sym, t0))
    dphi_dt0 = float(dphi_dt_sym.subs(t_sym, t0))
    
    # Радиальная и трансверсальная скорости
    Vr = dr_dt0
    Vphi = r0 * dphi_dt0
    V_mod = np.hypot(Vr, Vphi)
    
    # Ускорения
    d2r_dt2_0 = float(d2r_dt2_sym.subs(t_sym, t0))
    d2phi_dt2_0 = float(d2phi_dt2_sym.subs(t_sym, t0))
    
    # Радиальная и трансверсальная составляющие ускорения
    # ar = d²r/dt² - r·(dφ/dt)²
    ar = d2r_dt2_0 - r0 * dphi_dt0**2
    # aφ = r·d²φ/dt² + 2·dr/dt·dφ/dt
    aphi = r0 * d2phi_dt2_0 + 2 * dr_dt0 * dphi_dt0
    a_mod = np.hypot(ar, aphi)
    
    # Радиус кривизны (из декартовых координат)
    vx0 = float(sp.diff(x_expr, t_sym).subs(t_sym, t0))
    vy0 = float(sp.diff(y_expr, t_sym).subs(t_sym, t0))
    ax0 = float(sp.diff(x_expr, t_sym, 2).subs(t_sym, t0))
    ay0 = float(sp.diff(y_expr, t_sym, 2).subs(t_sym, t0))
    
    cross_2d = abs(vx0 * ay0 - vy0 * ax0)
    if cross_2d > 1e-10:
        rho = V_mod**3 / cross_2d
    else:
        rho = float('inf')
    
    return {
        't': t0,
        'x': x0,
        'y': y0,
        'r': r0,
        'phi': phi0,
        'dr_dt': dr_dt0,
        'dphi_dt': dphi_dt0,
        'd2r_dt2': d2r_dt2_0,
        'd2phi_dt2': d2phi_dt2_0,
        'Vr': Vr,
        'Vphi': Vphi,
        'V_mod': V_mod,
        'ar': ar,
        'aphi': aphi,
        'a_mod': a_mod,
        'rho': rho
    }


def get_formulas():
    """Возвращает LaTeX-формулы для отображения."""
    return {
        'r': sp.latex(r_sym),
        'phi': sp.latex(phi_sym),
        'dr_dt': sp.latex(dr_dt_sym),
        'dphi_dt': sp.latex(dphi_dt_sym),
        'd2r_dt2': sp.latex(d2r_dt2_sym),
        'd2phi_dt2': sp.latex(d2phi_dt2_sym),
    }


def psk_compute_complex_motion(t=1):
    """Основная функция для ПСК."""
    vals = compute_values_at_time(t)
    
    data = {
        't': t,
        'point': np.array([vals['x'], vals['y']]),
        'r': vals['r'],
        'phi': vals['phi'],
        'dr_dt': vals['dr_dt'],
        'dphi_dt': vals['dphi_dt'],
        'd2r_dt2': vals['d2r_dt2'],
        'd2phi_dt2': vals['d2phi_dt2'],
        'Vr': vals['Vr'],
        'Vphi': vals['Vphi'],
        'V_mod': vals['V_mod'],
        'ar': vals['ar'],
        'aphi': vals['aphi'],
        'a_mod': vals['a_mod'],
        'rho': vals['rho']
    }
    
    return data, get_formulas()
