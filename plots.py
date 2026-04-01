from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import calc


PLOT_REGISTRY = {
    # Пример 1
    "ex1_traj": "Пример 1: траектория",
    "ex1_coords": "Пример 1: координаты",
    "ex1_va": "Пример 1: |v| и |a|",
    "ex1_tn": "Пример 1: a_tau и a_n",
    "ex1_rho": "Пример 1: rho(t)",
    "ex1_vvec": "Пример 1: векторная диаграмма скорости (t0)",
    "ex1_avec": "Пример 1: векторная диаграмма ускорения (t0)",

    # Пример 2
    "ex2_traj3d": "Пример 2: 3D траектория",
    "ex2_coords": "Пример 2: координаты",
    "ex2_va": "Пример 2: |v| и |a|",
    "ex2_tn": "Пример 2: a_tau и a_n",
    "ex2_rho": "Пример 2: rho(t)",

    # Пример 3
    "ex3_traj": "Пример 3: траектория",
    "ex3_inputs": "Пример 3: r(t), φ'(t), Ω(t)",
    "ex3_va": "Пример 3: |v| и |a|",
    "ex3_vdecomp": "Пример 3: разложение скоростей",
    "ex3_ardecomp": "Пример 3: разложение a_r",
    "ex3_apdecomp": "Пример 3: разложение a_φ",
    "ex3_tn": "Пример 3: a_tau и a_n",
    "ex3_rho": "Пример 3: rho(t)",
    "ex3_vvec": "Пример 3: векторная диаграмма скорости (t0)",
    "ex3_avec": "Пример 3: векторная диаграмма ускорения (t0)",
}


def _vector_diagram(vectors, labels, title):
    fig = plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)

    pts = np.array([[0, 0]] + [v for v in vectors] + [np.sum(np.array(vectors), axis=0)])
    limx = np.max(np.abs(pts[:, 0])) + 0.6
    limy = np.max(np.abs(pts[:, 1])) + 0.6
    ax.set_xlim(-limx, limx)
    ax.set_ylim(-limy, limy)

    for vec, lab in zip(vectors, labels):
        ax.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1)
        ax.text(vec[0], vec[1], lab)

    res = np.sum(np.array(vectors), axis=0)
    ax.quiver(0, 0, res[0], res[1], angles='xy', scale_units='xy', scale=1)
    ax.text(res[0], res[1], 'Σ')

    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    return fig


def make_plot(name: str):
    # Пример 1
    if name.startswith("ex1_"):
        p = calc.ex1_params()
        arr = calc.ex1_arrays(p["u"], p["omega"], p["T"])
        pt = calc.ex1_point(p["u"], p["omega"], p["t0"])

        if name == "ex1_traj":
            fig = plt.figure()
            plt.plot(arr["x"], arr["y"])
            plt.axis('equal')
            plt.grid(True)
            plt.xlabel('x, м'); plt.ylabel('y, м')
            plt.title('Пример 1: траектория (спираль Архимеда)')
            return fig

        if name == "ex1_coords":
            fig = plt.figure()
            plt.plot(arr["t"], arr["x"], label='x(t)')
            plt.plot(arr["t"], arr["y"], label='y(t)')
            plt.grid(True); plt.legend();
            plt.xlabel('t, c'); plt.ylabel('м')
            plt.title('Пример 1: координаты')
            return fig

        if name == "ex1_va":
            fig = plt.figure()
            plt.plot(arr["t"], arr["v"], label='|v|')
            plt.plot(arr["t"], arr["a"], label='|a|')
            plt.grid(True); plt.legend();
            plt.xlabel('t, c'); plt.ylabel('СИ')
            plt.title('Пример 1: модули скорости и ускорения')
            return fig

        if name == "ex1_tn":
            fig = plt.figure()
            plt.plot(arr["t"], arr["a_tau"], label='a_τ')
            plt.plot(arr["t"], arr["a_n"], label='a_n')
            plt.grid(True); plt.legend();
            plt.xlabel('t, c'); plt.ylabel('м/с²')
            plt.title('Пример 1: касательное и нормальное ускорения')
            return fig

        if name == "ex1_rho":
            fig = plt.figure()
            plt.plot(arr["t"], arr["rho"])
            plt.grid(True)
            plt.xlabel('t, c'); plt.ylabel('ρ, м')
            plt.title('Пример 1: радиус кривизны ρ(t)')
            return fig

        if name == "ex1_vvec":
            # v = v_r + v_phi
            v_r = p["u"] * pt["er"]
            v_phi = pt["v_phi"] * pt["ephi"]
            return _vector_diagram([v_r, v_phi], ['v_r', 'v_φ'], f"Пример 1: скорость при t={p['t0']} c")

        if name == "ex1_avec":
            a_r = pt["a_r"] * pt["er"]
            a_phi = pt["a_phi"] * pt["ephi"]
            return _vector_diagram([a_r, a_phi], ['a_r', 'a_φ'], f"Пример 1: ускорение при t={p['t0']} c")

    # Пример 2
    if name.startswith("ex2_"):
        p = calc.ex2_params()
        arr = calc.ex2_arrays(p["V"], p["R"], p["Omega"], p["alpha"], p["h"], p["T"])

        if name == "ex2_traj3d":
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(arr["x"], arr["y"], arr["z"])
            ax.set_xlabel('x, м'); ax.set_ylabel('y, м'); ax.set_zlabel('z, м')
            ax.set_title('Пример 2: 3D траектория')
            return fig

        if name == "ex2_coords":
            fig = plt.figure()
            plt.plot(arr["t"], arr["x"], label='x(t)')
            plt.plot(arr["t"], arr["y"], label='y(t)')
            plt.plot(arr["t"], arr["z"], label='z(t)')
            plt.grid(True); plt.legend();
            plt.xlabel('t, c'); plt.ylabel('м')
            plt.title('Пример 2: координаты')
            return fig

        if name == "ex2_va":
            fig = plt.figure()
            plt.plot(arr["t"], arr["v"], label='|v|')
            plt.plot(arr["t"], arr["a"], label='|a|')
            plt.grid(True); plt.legend();
            plt.xlabel('t, c'); plt.ylabel('СИ')
            plt.title('Пример 2: модули скорости и ускорения')
            return fig

        if name == "ex2_tn":
            fig = plt.figure()
            plt.plot(arr["t"], arr["a_tau"], label='a_τ')
            plt.plot(arr["t"], arr["a_n"], label='a_n')
            plt.grid(True); plt.legend();
            plt.xlabel('t, c'); plt.ylabel('м/с²')
            plt.title('Пример 2: касательное и нормальное ускорения')
            return fig

        if name == "ex2_rho":
            fig = plt.figure()
            plt.plot(arr["t"], arr["rho"])
            plt.grid(True)
            plt.xlabel('t, c'); plt.ylabel('ρ, м')
            plt.title('Пример 2: радиус кривизны ρ(t)')
            return fig

    # Пример 3
    if name.startswith("ex3_"):
        p = calc.ex3_params()
        arr = calc.ex3_arrays(p["Omega0"], p["beta"], p["T"])
        pt = calc.ex3_point(p["Omega0"], p["beta"], p["t0"])

        if name == "ex3_traj":
            fig = plt.figure()
            plt.plot(arr["x"], arr["y"])
            plt.axis('equal')
            plt.grid(True)
            plt.xlabel('x, м'); plt.ylabel('y, м')
            plt.title('Пример 3: абсолютная траектория')
            return fig

        if name == "ex3_inputs":
            fig = plt.figure()
            plt.plot(arr["t"], arr["r"], label='r(t)')
            plt.plot(arr["t"], arr["phi_rel"], label="φ'(t)")
            plt.plot(arr["t"], arr["Omega"], label='Ω(t)')
            plt.grid(True); plt.legend();
            plt.xlabel('t, c'); plt.ylabel('СИ')
            plt.title("Пример 3: исходные функции")
            return fig

        if name == "ex3_va":
            fig = plt.figure()
            plt.plot(arr["t"], arr["v"], label='|v|')
            plt.plot(arr["t"], arr["a"], label='|a|')
            plt.grid(True); plt.legend();
            plt.xlabel('t, c'); plt.ylabel('СИ')
            plt.title('Пример 3: модули скорости и ускорения')
            return fig

        if name == "ex3_vdecomp":
            fig = plt.figure()
            plt.plot(arr["t"], arr["v_phi_rel"], label="v_φ(rel)")
            plt.plot(arr["t"], arr["v_phi_e"], label="v_φ(перен)")
            plt.plot(arr["t"], arr["v_phi"], label="v_φ(abs)")
            plt.plot(arr["t"], arr["v_r"], label="v_r")
            plt.grid(True); plt.legend();
            plt.xlabel('t, c'); plt.ylabel('м/с')
            plt.title('Пример 3: разложение скоростей')
            return fig

        if name == "ex3_ardecomp":
            fig = plt.figure()
            plt.plot(arr["t"], arr["a_r_rel"], label="a_r(rel)")
            plt.plot(arr["t"], arr["a_r_e"], label="a_r(e)")
            plt.plot(arr["t"], arr["a_r_k"], label="a_r(k)")
            plt.plot(arr["t"], arr["a_r"], label="a_r(abs)")
            plt.grid(True); plt.legend();
            plt.xlabel('t, c'); plt.ylabel('м/с²')
            plt.title('Пример 3: разложение a_r')
            return fig

        if name == "ex3_apdecomp":
            fig = plt.figure()
            plt.plot(arr["t"], arr["a_phi_rel"], label="a_φ(rel)")
            plt.plot(arr["t"], arr["a_phi_e"], label="a_φ(e)")
            plt.plot(arr["t"], arr["a_phi_k"], label="a_φ(k)")
            plt.plot(arr["t"], arr["a_phi"], label="a_φ(abs)")
            plt.grid(True); plt.legend();
            plt.xlabel('t, c'); plt.ylabel('м/с²')
            plt.title('Пример 3: разложение a_φ')
            return fig

        if name == "ex3_tn":
            fig = plt.figure()
            plt.plot(arr["t"], arr["a_tau"], label='a_τ')
            plt.plot(arr["t"], arr["a_n"], label='a_n')
            plt.grid(True); plt.legend();
            plt.xlabel('t, c'); plt.ylabel('м/с²')
            plt.title('Пример 3: касательное и нормальное ускорения')
            return fig

        if name == "ex3_rho":
            fig = plt.figure()
            plt.plot(arr["t"], arr["rho"])
            plt.grid(True)
            plt.xlabel('t, c'); plt.ylabel('ρ, м')
            plt.title('Пример 3: радиус кривизны ρ(t)')
            return fig

        if name == "ex3_vvec":
            v_r = pt["v_r"] * pt["er"]
            v_phi_rel = pt["v_phi_rel"] * pt["ephi"]
            v_phi_e = pt["v_phi_e"] * pt["ephi"]
            return _vector_diagram([v_r, v_phi_rel, v_phi_e], ['v_r', "v_φ'", 'v_φe'], f"Пример 3: скорость при t={p['t0']} c")

        if name == "ex3_avec":
            return _vector_diagram([pt["a_rel_vec"], pt["a_e_vec"], pt["a_k_vec"]], ['a_rel', 'a_e', 'a_k'], f"Пример 3: ускорение (слагаемые) при t={p['t0']} c")

    raise ValueError(f"Unknown plot name: {name}")
