from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Row:
    name: str
    value: float
    unit: str
    note: str = ""


def sig(x: float, n: int = 3) -> str:
    """Форматирование до n значащих цифр."""
    if x == 0:
        return "0"
    return f"{x:.{n}g}"


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def tangential_normal(v: np.ndarray, a: np.ndarray) -> Tuple[float, float, float, float]:
    """Возвращает (|v|, |a|, a_tau, a_n) по формулам:
    a_tau = (v·a)/|v|, a_n = |v×a|/|v|.
    Для плоского случая (2D) считаем псевдоскалярное произведение как z‑компоненту.
    """
    vmag = norm(v)
    amag = norm(a)
    if vmag < 1e-12:
        return vmag, amag, float("nan"), float("nan")

    a_tau = float(np.dot(v, a)) / vmag

    if v.shape[0] == 2:
        cross = v[0] * a[1] - v[1] * a[0]
        cross_mag = abs(float(cross))
    else:
        cross_mag = norm(np.cross(v, a))

    a_n = cross_mag / vmag
    return vmag, amag, a_tau, a_n


def curvature_radius(vmag: float, a_n: float) -> Tuple[float, float]:
    """κ = a_n / v^2, ρ = 1/κ."""
    if vmag < 1e-12:
        return float("nan"), float("nan")
    kappa = a_n / (vmag**2)
    rho = 1.0 / kappa if abs(kappa) > 1e-12 else float("inf")
    return kappa, rho


# -----------------------------
# Пример 1: спираль Архимеда
# -----------------------------

def ex1_params() -> Dict[str, float]:
    return {"u": 0.8, "omega": 1.5, "t0": 2.0, "T": 6.0}


def ex1_arrays(u: float, omega: float, T: float, n: int = 1200):
    t = np.linspace(0.0, T, n)
    r = u * t
    phi = omega * t
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    # Полярные компоненты
    v_r = np.full_like(t, u)
    v_phi = r * omega
    v = np.sqrt(v_r**2 + v_phi**2)

    a_r = -r * omega**2
    a_phi = 2.0 * u * omega * np.ones_like(t)
    a = np.sqrt(a_r**2 + a_phi**2)

    # Касательная/нормаль через производные
    a_tau = np.gradient(v, t)
    a_n = np.sqrt(np.maximum(a**2 - a_tau**2, 0.0))

    kappa = a_n / np.maximum(v**2, 1e-12)
    rho = 1.0 / np.maximum(kappa, 1e-12)

    return {
        "t": t,
        "x": x,
        "y": y,
        "r": r,
        "phi": phi,
        "v": v,
        "a": a,
        "a_tau": a_tau,
        "a_n": a_n,
        "rho": rho,
    }


def ex1_point(u: float, omega: float, t0: float) -> Dict[str, float]:
    r = u * t0
    phi = omega * t0
    x = r * math.cos(phi)
    y = r * math.sin(phi)

    # v = u e_r + r ω e_phi
    er = np.array([math.cos(phi), math.sin(phi)])
    ephi = np.array([-math.sin(phi), math.cos(phi)])
    vvec = u * er + (r * omega) * ephi
    # a = -(r ω^2) e_r + 2uω e_phi
    avec = (-(r * omega**2)) * er + (2 * u * omega) * ephi

    vmag, amag, a_tau, a_n = tangential_normal(vvec, avec)
    kappa, rho = curvature_radius(vmag, a_n)

    return {
        "x": x,
        "y": y,
        "r": r,
        "phi": phi,
        "v_r": u,
        "v_phi": r * omega,
        "v": vmag,
        "a_r": -(r * omega**2),
        "a_phi": 2 * u * omega,
        "a": amag,
        "a_tau": a_tau,
        "a_n": a_n,
        "kappa": kappa,
        "rho": rho,
        "vvec": vvec,
        "avec": avec,
        "er": er,
        "ephi": ephi,
    }


# -----------------------------
# Пример 2: пространственный
# -----------------------------

def ex2_params() -> Dict[str, float]:
    return {"V": 1.0, "R": 0.5, "Omega": 1.5, "alpha": 2.5, "h": 0.2, "t0": 0.0, "T": 8.0}


def ex2_arrays(V: float, R: float, Omega: float, alpha: float, h: float, T: float, n: int = 1500):
    gamma = alpha + Omega
    t = np.linspace(0.0, T, n)
    x = V * t + R * np.cos(gamma * t)
    y = R * np.sin(gamma * t)
    z = h * t

    # Производные численно (для графиков)
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    vz = np.gradient(z, t)
    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)
    az = np.gradient(vz, t)

    v = np.sqrt(vx**2 + vy**2 + vz**2)
    a = np.sqrt(ax**2 + ay**2 + az**2)

    a_tau = (vx * ax + vy * ay + vz * az) / np.maximum(v, 1e-12)
    cross = np.cross(np.vstack([vx, vy, vz]).T, np.vstack([ax, ay, az]).T)
    a_n = np.linalg.norm(cross, axis=1) / np.maximum(v, 1e-12)

    kappa = a_n / np.maximum(v**2, 1e-12)
    rho = 1.0 / np.maximum(kappa, 1e-12)

    return {
        "t": t,
        "x": x,
        "y": y,
        "z": z,
        "v": v,
        "a": a,
        "a_tau": a_tau,
        "a_n": a_n,
        "rho": rho,
        "gamma": gamma,
    }


def ex2_point(V: float, R: float, Omega: float, alpha: float, h: float, t0: float) -> Dict[str, float]:
    gamma = alpha + Omega
    x = V * t0 + R * math.cos(gamma * t0)
    y = R * math.sin(gamma * t0)
    z = h * t0

    # Аналитические производные
    vx = V - R * gamma * math.sin(gamma * t0)
    vy = R * gamma * math.cos(gamma * t0)
    vz = h

    ax = -R * gamma**2 * math.cos(gamma * t0)
    ay = -R * gamma**2 * math.sin(gamma * t0)
    az = 0.0

    vvec = np.array([vx, vy, vz], dtype=float)
    avec = np.array([ax, ay, az], dtype=float)

    vmag, amag, a_tau, a_n = tangential_normal(vvec, avec)
    kappa, rho = curvature_radius(vmag, a_n)

    return {
        "x": x,
        "y": y,
        "z": z,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "ax": ax,
        "ay": ay,
        "az": az,
        "v": vmag,
        "a": amag,
        "a_tau": a_tau,
        "a_n": a_n,
        "kappa": kappa,
        "rho": rho,
        "gamma": gamma,
        "vvec": vvec,
        "avec": avec,
    }


# -----------------------------
# Пример 3: полярная система во вращающейся
# -----------------------------

def ex3_params() -> Dict[str, float]:
    return {"Omega0": 1.0, "beta": 0.3, "t0": 1.0, "T": 6.0}


def ex3_arrays(Omega0: float, beta: float, T: float, n: int = 1500):
    t = np.linspace(0.0, T, n)

    r = 1.0 + 0.2 * np.sin(2.0 * t)
    rd = 0.4 * np.cos(2.0 * t)
    rdd = -0.8 * np.sin(2.0 * t)

    phi_rel = 0.5 * t
    phi_rel_d = 0.5 * np.ones_like(t)
    phi_rel_dd = np.zeros_like(t)

    Omega = Omega0 + beta * t
    Omega_d = beta * np.ones_like(t)

    Phi = Omega0 * t + 0.5 * beta * t**2

    phi = phi_rel + Phi
    phi_d = phi_rel_d + Omega
    phi_dd = phi_rel_dd + Omega_d

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    # Скорость в полярных компонентах
    v_r = rd
    v_phi = r * phi_d
    v = np.sqrt(v_r**2 + v_phi**2)

    # Ускорение (абсолютные полярные компоненты)
    a_r = rdd - r * (phi_d**2)
    a_phi = r * phi_dd + 2.0 * rd * phi_d
    a = np.sqrt(a_r**2 + a_phi**2)

    a_tau = np.gradient(v, t)
    a_n = np.sqrt(np.maximum(a**2 - a_tau**2, 0.0))

    kappa = a_n / np.maximum(v**2, 1e-12)
    rho = 1.0 / np.maximum(kappa, 1e-12)

    # Разложения (rel / e / k) для графиков
    v_phi_rel = r * phi_rel_d
    v_phi_e = r * Omega

    a_r_rel = rdd - r * (phi_rel_d**2)
    a_phi_rel = r * phi_rel_dd + 2.0 * rd * phi_rel_d

    a_r_e = -r * (Omega**2)
    a_phi_e = r * Omega_d

    a_r_k = -2.0 * Omega * r * phi_rel_d
    a_phi_k = 2.0 * Omega * rd

    return {
        "t": t,
        "x": x,
        "y": y,
        "r": r,
        "phi_rel": phi_rel,
        "Omega": Omega,
        "v": v,
        "a": a,
        "a_tau": a_tau,
        "a_n": a_n,
        "rho": rho,
        "v_r": v_r,
        "v_phi": v_phi,
        "v_phi_rel": v_phi_rel,
        "v_phi_e": v_phi_e,
        "a_r": a_r,
        "a_phi": a_phi,
        "a_r_rel": a_r_rel,
        "a_phi_rel": a_phi_rel,
        "a_r_e": a_r_e,
        "a_phi_e": a_phi_e,
        "a_r_k": a_r_k,
        "a_phi_k": a_phi_k,
        "phi": phi,
    }


def ex3_point(Omega0: float, beta: float, t0: float) -> Dict[str, float]:
    r = 1.0 + 0.2 * math.sin(2.0 * t0)
    rd = 0.4 * math.cos(2.0 * t0)
    rdd = -0.8 * math.sin(2.0 * t0)

    phi_rel = 0.5 * t0
    phi_rel_d = 0.5
    phi_rel_dd = 0.0

    Omega = Omega0 + beta * t0
    Omega_d = beta

    Phi = Omega0 * t0 + 0.5 * beta * t0**2

    phi = phi_rel + Phi
    phi_d = phi_rel_d + Omega
    phi_dd = phi_rel_dd + Omega_d

    x = r * math.cos(phi)
    y = r * math.sin(phi)

    # Скорости
    v_r = rd
    v_phi_rel = r * phi_rel_d
    v_phi_e = r * Omega
    v_phi = r * phi_d

    # Ускорения (rel/e/k)
    a_r_rel = rdd - r * (phi_rel_d**2)
    a_phi_rel = r * phi_rel_dd + 2.0 * rd * phi_rel_d

    a_r_e = -r * (Omega**2)
    a_phi_e = r * Omega_d

    a_r_k = -2.0 * Omega * r * phi_rel_d
    a_phi_k = 2.0 * Omega * rd

    a_r = a_r_rel + a_r_e + a_r_k
    a_phi = a_phi_rel + a_phi_e + a_phi_k

    # Перевод в декартовы для a_tau, a_n
    er = np.array([math.cos(phi), math.sin(phi)])
    ephi = np.array([-math.sin(phi), math.cos(phi)])

    vvec = v_r * er + v_phi * ephi
    avec = a_r * er + a_phi * ephi

    vmag, amag, a_tau, a_n = tangential_normal(vvec, avec)
    kappa, rho = curvature_radius(vmag, a_n)

    return {
        "x": x,
        "y": y,
        "r": r,
        "rd": rd,
        "rdd": rdd,
        "phi_rel": phi_rel,
        "phi_rel_d": phi_rel_d,
        "Omega": Omega,
        "Omega_d": Omega_d,
        "Phi": Phi,
        "phi": phi,
        "phi_d": phi_d,
        "phi_dd": phi_dd,
        "v_r": v_r,
        "v_phi_rel": v_phi_rel,
        "v_phi_e": v_phi_e,
        "v_phi": v_phi,
        "v": vmag,
        "a_r_rel": a_r_rel,
        "a_phi_rel": a_phi_rel,
        "a_r_e": a_r_e,
        "a_phi_e": a_phi_e,
        "a_r_k": a_r_k,
        "a_phi_k": a_phi_k,
        "a_r": a_r,
        "a_phi": a_phi,
        "a": amag,
        "a_tau": a_tau,
        "a_n": a_n,
        "kappa": kappa,
        "rho": rho,
        "vvec": vvec,
        "avec": avec,
        "er": er,
        "ephi": ephi,
        # отдельные векторы ускорений (для векторной диаграммы)
        "a_rel_vec": a_r_rel * er + a_phi_rel * ephi,
        "a_e_vec": a_r_e * er + a_phi_e * ephi,
        "a_k_vec": a_r_k * er + a_phi_k * ephi,
    }


# -----------------------------
# Контекст для HTML
# -----------------------------

def build_context() -> Dict:
    p1 = ex1_params()
    p2 = ex2_params()
    p3 = ex3_params()

    ex1p = ex1_point(p1["u"], p1["omega"], p1["t0"])
    ex2p = ex2_point(p2["V"], p2["R"], p2["Omega"], p2["alpha"], p2["h"], p2["t0"])
    ex3p = ex3_point(p3["Omega0"], p3["beta"], p3["t0"])

    # Таблицы (контрольные точки)
    ex1_rows = [
        Row("u", p1["u"], "м/с"),
        Row("ω", p1["omega"], "рад/с"),
        Row("t0", p1["t0"], "с"),
        Row("r(t0)=u t0", ex1p["r"], "м"),
        Row("φ(t0)=ω t0", ex1p["phi"], "рад"),
        Row("x(t0)", ex1p["x"], "м"),
        Row("y(t0)", ex1p["y"], "м"),
        Row("v_r", ex1p["v_r"], "м/с"),
        Row("v_φ=rω", ex1p["v_phi"], "м/с"),
        Row("|v|", ex1p["v"], "м/с"),
        Row("a_r=-rω²", ex1p["a_r"], "м/с²"),
        Row("a_φ=2uω", ex1p["a_phi"], "м/с²"),
        Row("|a|", ex1p["a"], "м/с²"),
        Row("a_τ", ex1p["a_tau"], "м/с²"),
        Row("a_n", ex1p["a_n"], "м/с²"),
        Row("κ=a_n/|v|²", ex1p["kappa"], "1/м"),
        Row("ρ=1/κ", ex1p["rho"], "м"),
    ]

    ex2_rows = [
        Row("V", p2["V"], "м/с"),
        Row("R", p2["R"], "м"),
        Row("Ω", p2["Omega"], "рад/с"),
        Row("α", p2["alpha"], "рад/с"),
        Row("h", p2["h"], "м/с"),
        Row("γ=α+Ω", ex2p["gamma"], "рад/с"),
        Row("t0", p2["t0"], "с"),
        Row("x(t0)", ex2p["x"], "м"),
        Row("y(t0)", ex2p["y"], "м"),
        Row("z(t0)", ex2p["z"], "м"),
        Row("v_x(t0)", ex2p["vx"], "м/с"),
        Row("v_y(t0)", ex2p["vy"], "м/с"),
        Row("v_z(t0)", ex2p["vz"], "м/с"),
        Row("|v|(t0)", ex2p["v"], "м/с"),
        Row("a_x(t0)", ex2p["ax"], "м/с²"),
        Row("a_y(t0)", ex2p["ay"], "м/с²"),
        Row("a_z(t0)", ex2p["az"], "м/с²"),
        Row("|a|(t0)", ex2p["a"], "м/с²"),
        Row("a_τ(t0)", ex2p["a_tau"], "м/с²"),
        Row("a_n(t0)", ex2p["a_n"], "м/с²"),
        Row("κ(t0)", ex2p["kappa"], "1/м"),
        Row("ρ(t0)", ex2p["rho"], "м"),
    ]

    ex3_rows = [
        Row("Ω0", p3["Omega0"], "рад/с"),
        Row("β", p3["beta"], "рад/с²"),
        Row("t0", p3["t0"], "с"),
        Row("Ω(t0)=Ω0+βt0", ex3p["Omega"], "рад/с"),
        Row("dΩ/dt", ex3p["Omega_d"], "рад/с²"),
        Row("r(t0)", ex3p["r"], "м"),
        Row("ṙ(t0)", ex3p["rd"], "м/с"),
        Row("r̈(t0)", ex3p["rdd"], "м/с²"),
        Row("φ'(t0)", ex3p["phi_rel"], "рад"),
        Row("φ̇'(t0)", ex3p["phi_rel_d"], "рад/с"),
        Row("Φ(t0)=∫Ωdt", ex3p["Phi"], "рад"),
        Row("φ(t0)=φ'+Φ", ex3p["phi"], "рад"),
        Row("φ̇(t0)=φ̇'+Ω", ex3p["phi_d"], "рад/с"),
        Row("φ̈(t0)=dΩ/dt", ex3p["phi_dd"], "рад/с²"),
        Row("x(t0)", ex3p["x"], "м"),
        Row("y(t0)", ex3p["y"], "м"),
        Row("v_r", ex3p["v_r"], "м/с"),
        Row("v_φ'(rel)", ex3p["v_phi_rel"], "м/с"),
        Row("v_φe(перен)", ex3p["v_phi_e"], "м/с"),
        Row("v_φ(abs)", ex3p["v_phi"], "м/с"),
        Row("|v|(abs)", ex3p["v"], "м/с"),
        Row("a_r(rel)", ex3p["a_r_rel"], "м/с²"),
        Row("a_φ(rel)", ex3p["a_phi_rel"], "м/с²"),
        Row("a_r(e)", ex3p["a_r_e"], "м/с²"),
        Row("a_φ(e)", ex3p["a_phi_e"], "м/с²"),
        Row("a_r(k)", ex3p["a_r_k"], "м/с²"),
        Row("a_φ(k)", ex3p["a_phi_k"], "м/с²"),
        Row("a_r(abs)", ex3p["a_r"], "м/с²"),
        Row("a_φ(abs)", ex3p["a_phi"], "м/с²"),
        Row("|a|(abs)", ex3p["a"], "м/с²"),
        Row("a_τ(abs)", ex3p["a_tau"], "м/с²"),
        Row("a_n(abs)", ex3p["a_n"], "м/с²"),
        Row("κ(abs)", ex3p["kappa"], "1/м"),
        Row("ρ(abs)", ex3p["rho"], "м"),
    ]

    summary = [
        {
            "case": "Пример 1",
            "moment": "t=2.00 c",
            "x": ex1p["x"],
            "y": ex1p["y"],
            "z": 0.0,
            "v": ex1p["v"],
            "a": ex1p["a"],
            "a_tau": ex1p["a_tau"],
            "a_n": ex1p["a_n"],
            "kappa": ex1p["kappa"],
            "rho": ex1p["rho"],
        },
        {
            "case": "Пример 2",
            "moment": "t=0.00 c",
            "x": ex2p["x"],
            "y": ex2p["y"],
            "z": ex2p["z"],
            "v": ex2p["v"],
            "a": ex2p["a"],
            "a_tau": ex2p["a_tau"],
            "a_n": ex2p["a_n"],
            "kappa": ex2p["kappa"],
            "rho": ex2p["rho"],
        },
        {
            "case": "Пример 3",
            "moment": "t=1.00 c",
            "x": ex3p["x"],
            "y": ex3p["y"],
            "z": 0.0,
            "v": ex3p["v"],
            "a": ex3p["a"],
            "a_tau": ex3p["a_tau"],
            "a_n": ex3p["a_n"],
            "kappa": ex3p["kappa"],
            "rho": ex3p["rho"],
        },
    ]

    def rows_to_dict(rows: List[Row]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for r in rows:
            out.append({
                "name": r.name,
                "value": sig(r.value, 3),
                "unit": r.unit,
                "note": r.note,
            })
        return out

    def summary_fmt(items: List[Dict]) -> List[Dict[str, str]]:
        out = []
        for it in items:
            out.append({
                "case": it["case"],
                "moment": it["moment"],
                "x": sig(it["x"], 3),
                "y": sig(it["y"], 3),
                "z": sig(it["z"], 3),
                "v": sig(it["v"], 3),
                "a": sig(it["a"], 3),
                "a_tau": sig(it["a_tau"], 3),
                "a_n": sig(it["a_n"], 3),
                "kappa": sig(it["kappa"], 3),
                "rho": sig(it["rho"], 3),
            })
        return out

    return {
        "today": date.today().isoformat(),
        "params": {"ex1": p1, "ex2": p2, "ex3": p3},
        "tables": {
            "ex1": rows_to_dict(ex1_rows),
            "ex2": rows_to_dict(ex2_rows),
            "ex3": rows_to_dict(ex3_rows),
            "summary": summary_fmt(summary),
        },
    }


def static_report_path() -> Path:
    # Файл лежит в exports/report_static.html относительно корня проекта
    here = Path(__file__).resolve().parent
    return here / "exports" / "report_static.html"
