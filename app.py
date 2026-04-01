from flask import Flask, render_template, send_file, make_response
import os
import io
import base64
import tempfile
import pdfkit
import pypandoc
import matplotlib.pyplot as plt
import numpy as np
from calc import compute_complex_motion
from plots import generate_all_plots

app = Flask(__name__)

# Создаём папку для статики, если её нет
os.makedirs('static', exist_ok=True)

def latex_to_png(latex_str):
    """Преобразует LaTeX-строку в base64-изображение PNG."""
    fig, ax = plt.subplots(figsize=(0.8, 0.4))
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.text(0.5, 0.5, f'${latex_str}$', fontsize=12, ha='center', va='center')
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', pad_inches=0.05, transparent=True)
    plt.close()
    img.seek(0)
    return base64.b64encode(img.read()).decode()

@app.route('/')
def index():
    data, formulas = compute_complex_motion(t=1)
    generate_all_plots(data)
    return render_template('report.html', data=data, formulas=formulas)

@app.route('/export/pdf')
def export_pdf():
    """Экспорт отчёта в PDF."""
    data, formulas = compute_complex_motion(t=1)
    generate_all_plots(data)

    # Преобразуем все формулы в изображения
    formula_images = {}
    for key, latex in formulas.items():
        formula_images[key] = latex_to_png(latex)

    # Получаем абсолютные пути к графикам
    static_abs_path = os.path.abspath('static')
    img_files = {
        'trajectory': os.path.join(static_abs_path, 'trajectory.png'),
        'velocities': os.path.join(static_abs_path, 'velocities.png'),
        'accelerations': os.path.join(static_abs_path, 'accelerations.png'),
    }

    rendered = render_template('report_export.html', data=data, formulas=formulas,
                               formula_images=formula_images, images=img_files, export=True)

    # Сохраняем HTML во временный файл
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
        f.write(rendered.encode('utf-8'))
        html_path = f.name

    pdf_path = html_path.replace('.html', '.pdf')
    try:
        pdfkit.from_file(html_path, pdf_path, options={'enable-local-file-access': None})
    except Exception as e:
        os.unlink(html_path)
        return f"Ошибка генерации PDF: {e}", 500

    with open(pdf_path, 'rb') as f:
        pdf_data = f.read()
    os.unlink(html_path)
    os.unlink(pdf_path)

    return send_file(
        io.BytesIO(pdf_data),
        as_attachment=True,
        download_name='report.pdf',
        mimetype='application/pdf'
    )

@app.route('/export/word')
def export_word():
    """Экспорт отчёта в Word (docx)."""
    data, formulas = compute_complex_motion(t=1)
    generate_all_plots(data)

    formula_images = {}
    for key, latex in formulas.items():
        formula_images[key] = latex_to_png(latex)

    static_abs_path = os.path.abspath('static')
    img_files = {
        'trajectory': os.path.join(static_abs_path, 'trajectory.png'),
        'velocities': os.path.join(static_abs_path, 'velocities.png'),
        'accelerations': os.path.join(static_abs_path, 'accelerations.png'),
    }

    rendered = render_template('report_export.html', data=data, formulas=formulas,
                               formula_images=formula_images, images=img_files, export=True)

    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f_html:
        f_html.write(rendered.encode('utf-8'))
        html_path = f_html.name

    with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as f_docx:
        docx_path = f_docx.name

    try:
        pypandoc.convert_file(html_path, 'docx', outputfile=docx_path)
        with open(docx_path, 'rb') as f:
            docx_data = f.read()
    except Exception as e:
        os.unlink(html_path)
        if os.path.exists(docx_path):
            os.unlink(docx_path)
        return f"Ошибка генерации Word: {e}", 500

    os.unlink(html_path)
    os.unlink(docx_path)

    return send_file(
        io.BytesIO(docx_data),
        as_attachment=True,
        download_name='report.docx',
        mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
