import numpy as np
import sympy as sp
from scipy import integrate


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

# Скорости в ДСК
vx_expr = sp.diff(x_expr, t_sym).simplify()
vy_expr = sp.diff(y_expr, t_sym).simplify()

# Ускорения в ДСК
ax_expr = sp.diff(vx_expr, t_sym).simplify()
ay_expr = sp.diff(vy_expr, t_sym).simplify()

# Модуль скорости (символьный)
V_sym = sp.sqrt(vx_expr**2 + vy_expr**2).simplify()


def compute_values_at_time(t0=1):
    """Вычисляет все кинематические величины в момент времени t0 для ЕСЗД."""
    # Координаты
    x0 = float(x_expr.subs(t_sym, t0))
    y0 = float(y_expr.subs(t_sym, t0))
    
    # Скорости
    vx0 = float(vx_expr.subs(t_sym, t0))
    vy0 = float(vy_expr.subs(t_sym, t0))
    
    # Ускорения
    ax0 = float(ax_expr.subs(t_sym, t0))
    ay0 = float(ay_expr.subs(t_sym, t0))
    
    # Модуль скорости (полная скорость)
    V = np.hypot(vx0, vy0)
    
    # Касательное ускорение (производная модуля скорости по времени)
    # a_tau = dV/dt = (vx*ax + vy*ay) / V
    a_tau = (vx0 * ax0 + vy0 * ay0) / V if V > 0 else 0
    
    # Нормальное ускорение (центростремительное)
    # a_n = sqrt(a^2 - a_tau^2)
    a_mod = np.hypot(ax0, ay0)
    a_n = np.sqrt(max(0, a_mod**2 - a_tau**2))
    
    # Радиус кривизны
    # ρ = V^2 / a_n (если a_n > 0)
    if a_n > 1e-10:
        rho = V**2 / a_n
    else:
        rho = float('inf')
    
    # Длина дуги (численное интегрирование)
    # Создаём функцию для численного интегрирования модуля скорости
    def V_numeric(t):
        t_val = float(t)
        vx = float(vx_expr.subs(t_sym, t_val))
        vy = float(vy_expr.subs(t_sym, t_val))
        return np.hypot(vx, vy)
    
    # Вычисляем интеграл от 0 до t0
    s0, _ = integrate.quad(V_numeric, 0, t0, limit=100)
    
    # Единичные векторы касательной и нормали
    # Касательный вектор: τ = (vx, vy) / V
    tau_x = vx0 / V if V > 0 else 1
    tau_y = vy0 / V if V > 0 else 0
    
    # Нормальный вектор (повернутый касательный на 90°)
    # Определяем направление по знаку векторного произведения
    cross = vx0 * ay0 - vy0 * ax0
    if cross > 0:
        n_x = -tau_y
        n_y = tau_x
    else:
        n_x = tau_y
        n_y = -tau_x
    
    return {
        't': t0,
        'x': x0,
        'y': y0,
        'vx': vx0,
        'vy': vy0,
        'ax': ax0,
        'ay': ay0,
        'V': V,
        'a_tau': a_tau,
        'a_n': a_n,
        'a_mod': a_mod,
        'rho': rho,
        's': s0,
        'tau_x': tau_x,
        'tau_y': tau_y,
        'n_x': n_x,
        'n_y': n_y
    }


def get_formulas():
    """Возвращает LaTeX-формулы для отображения."""
    # Касательное ускорение (символьно)
    a_tau_sym = sp.diff(V_sym, t_sym).simplify()
    
    # Для нормального ускорения используем формулу a_n = V^2 / ρ
    cross_sym = vx_expr * ay_expr - vy_expr * ax_expr
    rho_sym = V_sym**3 / sp.Abs(cross_sym)
    a_n_sym = V_sym**2 / rho_sym
    
    return {
        'x': sp.latex(x_expr),
        'y': sp.latex(y_expr),
        'vx': sp.latex(vx_expr),
        'vy': sp.latex(vy_expr),
        'ax': sp.latex(ax_expr),
        'ay': sp.latex(ay_expr),
        'V': sp.latex(V_sym),
        'a_tau': sp.latex(a_tau_sym),
        'a_n': sp.latex(a_n_sym),
        'rho': sp.latex(rho_sym),
    }


def eszd_compute_complex_motion(t=1):
    """Основная функция для ЕСЗД."""
    vals = compute_values_at_time(t)
    
    data = {
        't': t,
        'point': np.array([vals['x'], vals['y']]),
        'V': vals['V'],
        'V_vec': np.array([vals['vx'], vals['vy']]),
        'a_tau': vals['a_tau'],
        'a_n': vals['a_n'],
        'a_mod': vals['a_mod'],
        'a_tau_vec': np.array([vals['tau_x'], vals['tau_y']]) * vals['a_tau'],
        'a_n_vec': np.array([vals['n_x'], vals['n_y']]) * vals['a_n'],
        'rho': vals['rho'],
        's': vals['s'],
        'tau': np.array([vals['tau_x'], vals['tau_y']]),
        'n': np.array([vals['n_x'], vals['n_y']])
    }
    
    return data, get_formulas()
