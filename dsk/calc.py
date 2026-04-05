import numpy as np
import sympy as sp


# Символьные переменные и выражения
t_sym = sp.Symbol('t', real=True)
a_sym, b_sym = sp.symbols('a b', real=True)

# Параметры движения
a_val = 8
b_val = -2

# Движение точки в ДСК
x_sym = a_sym * sp.cos(b_sym * sp.pi * t_sym**2 / 6)
y_sym = -2 * a_sym * sp.sin(b_sym * sp.pi * t_sym**2 / 6)

x_expr = x_sym.subs({a_sym: a_val, b_sym: b_val}).simplify()
y_expr = y_sym.subs({a_sym: a_val, b_sym: b_val}).simplify()

vx_expr = sp.diff(x_expr, t_sym).simplify()
vy_expr = sp.diff(y_expr, t_sym).simplify()
ax_expr = sp.diff(vx_expr, t_sym).simplify()
ay_expr = sp.diff(vy_expr, t_sym).simplify()


def compute_values_at_time(t0=1):
    """Вычисляет все кинематические величины в момент времени t0 для ДСК."""
    x0 = float(x_expr.subs(t_sym, t0))
    y0 = float(y_expr.subs(t_sym, t0))
    
    vx0 = float(vx_expr.subs(t_sym, t0))
    vy0 = float(vy_expr.subs(t_sym, t0))
    
    ax0 = float(ax_expr.subs(t_sym, t0))
    ay0 = float(ay_expr.subs(t_sym, t0))
    
    V_mod = np.hypot(vx0, vy0)
    a_mod = np.hypot(ax0, ay0)
    
    return {
        't': t0,
        'x': x0,
        'y': y0,
        'vx': vx0,
        'vy': vy0,
        'ax': ax0,
        'ay': ay0,
        'V_mod': V_mod,
        'a_mod': a_mod
    }


def get_formulas():
    """Возвращает LaTeX-формулы для отображения."""
    return {
        'x': sp.latex(x_expr),
        'y': sp.latex(y_expr),
        'dxdt': sp.latex(vx_expr),
        'dydt': sp.latex(vy_expr),
        'd2xdt2': sp.latex(ax_expr),
        'd2ydt2': sp.latex(ay_expr),
    }


# Для обратной совместимости с существующим кодом
def dsk_compute_complex_motion(t=1):
    vals = compute_values_at_time(t)
    
    data = {
        't': t,
        'point': np.array([vals['x'], vals['y'], 0.0]),
        'V_rel': np.array([vals['vx'], vals['vy'], 0.0]),
        'V_abs': np.array([vals['vx'], vals['vy'], 0.0]),
        'a_rel': np.array([vals['ax'], vals['ay'], 0.0]),
        'a_abs': np.array([vals['ax'], vals['ay'], 0.0]),
        'V_abs_mod': vals['V_mod'],
        'a_abs_mod': vals['a_mod'],
    }
    
    return data, get_formulas()
