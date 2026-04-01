
### app.py

#```python
from __future__ import annotations

import io
from flask import Flask, Response, abort, render_template, send_file

import calc
import plots

app = Flask(__name__)

# Простой кэш PNG в памяти: name -> bytes
_PLOT_CACHE: dict[str, bytes] = {}


@app.get("/")
def report():
    """Одностраничный отчёт."""
    context = calc.build_context()
    return render_template("report.html", **context)


@app.get("/plot/<name>.png")
def plot_png(name: str):
    """Динамическая генерация графиков Matplotlib."""
    if name not in plots.PLOT_REGISTRY:
        abort(404, description=f"Unknown plot: {name}")

    if name not in _PLOT_CACHE:
        fig = plots.make_plot(name)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        _PLOT_CACHE[name] = buf.getvalue()

    return Response(_PLOT_CACHE[name], mimetype="image/png")


@app.get("/download/static")
def download_static_report():
    """Скачивание офлайн‑версии отчёта (самодостаточный HTML)."""
    path = calc.static_report_path()
    return send_file(path, as_attachment=True, download_name="report_static.html")


if __name__ == "__main__":
    # debug=True удобно для разработки; для сдачи можно поставить False
    app.run(host="127.0.0.1", port=5000, debug=True)
