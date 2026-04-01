from flask import Flask, render_template, send_file
import os
import io
import base64
import tempfile
import pdfkit
import pypandoc
import matplotlib.pyplot as plt
import numpy as np
from calc import compute_complex_motion
from plots import generate_all_plots, generate_interactive_trajectory, generate_interactive_velocities, generate_interactive_accelerations

app = Flask(__name__)

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
    # Генерируем статичные PNG для экспорта (можно вызывать только при экспорте, но оставим здесь)
    generate_all_plots(data)
    # Интерактивные графики
    traj_json = generate_interactive_trajectory(data)
    vel_json = generate_interactive_velocities(data)
    acc_json = generate_interactive_accelerations(data)
    return render_template('report.html', data=data, formulas=formulas,
                           traj_json=traj_json, vel_json=vel_json, acc_json=acc_json)

def prepare_export_data():
    """Возвращает data, formulas и formula_images для экспорта."""
    data, formulas = compute_complex_motion(t=1)
    generate_all_plots(data)  # PNG для экспорта

    formula_images = {}
    for key, latex in formulas.items():
        formula_images[key] = latex_to_png(latex)

    extra_formulas = {
        'V_abs_eq': r'\mathbf{V}_{\text{абс}} = \mathbf{V}_{\text{отн}} + \mathbf{V}_{\text{пер,пост}} + \mathbf{V}_{\text{пер,вр}}',
        'V_rot_eq': r'\mathbf{V}_{\text{пер,вр}} = \boldsymbol{\omega} \times \mathbf{r}_{\text{отн}}',
        'a_abs_eq': r'\mathbf{a}_{\text{абс}} = \mathbf{a}_{\text{отн}} + \mathbf{a}_{\text{пер}} + \mathbf{a}_{\text{кор}}',
        'a_trans_eq': r'\mathbf{a}_{\text{пер}} = \mathbf{a}_{\text{пер,пост}} + \mathbf{a}_{\text{пер,ц}} + \mathbf{a}_{\text{пер,вр}}',
        'a_centr_eq': r'\mathbf{a}_{\text{пер,ц}} = -\omega^2 \mathbf{r}_{\text{отн}}',
        'a_rot_eq': r'\mathbf{a}_{\text{пер,вр}} = \boldsymbol{\alpha} \times \mathbf{r}_{\text{отн}}',
        'a_cor_eq': r'\mathbf{a}_{\text{кор}} = 2\,\boldsymbol{\omega} \times \mathbf{V}_{\text{отн}}',
        'rho_eq': r'\rho = \frac{V_{\text{абс}}^3}{|\mathbf{V}_{\text{абс}} \times \mathbf{a}_{\text{абс}}|}',
    }
    for key, latex in extra_formulas.items():
        formula_images[key] = latex_to_png(latex)

    static_abs_path = os.path.abspath('static')
    img_files = {
        'trajectory': os.path.join(static_abs_path, 'trajectory.png'),
        'velocities': os.path.join(static_abs_path, 'velocities.png'),
        'accelerations': os.path.join(static_abs_path, 'accelerations.png'),
    }
    return data, formulas, formula_images, img_files

@app.route('/export/pdf')
def export_pdf():
    data, formulas, formula_images, img_files = prepare_export_data()
    rendered = render_template('report_export.html', data=data, formulas=formulas,
                               formula_images=formula_images, images=img_files, export=True)

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
    data, formulas, formula_images, img_files = prepare_export_data()
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
