from flask import Flask, render_template, send_file, url_for
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from calc import compute_complex_motion
from plots import generate_interactive_trajectory, \
                  generate_interactive_velocities, generate_interactive_accelerations, \
                  generate_interactive_trajectory_with_velocities, \
                  generate_interactive_trajectory_with_accelerations
from export_utils import generate_export_pdf, generate_export_word

app = Flask(__name__)
os.makedirs('static', exist_ok=True)


# Вспомогательные функции
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


# Маршруты
@app.route('/')
def index():
    data, formulas = compute_complex_motion(t=1)
    
    # Генерируем JSON для интерактивных графиков
    traj_json = generate_interactive_trajectory(data)
    vel_json = generate_interactive_velocities(data)
    acc_json = generate_interactive_accelerations(data)
    traj_with_vel_json = generate_interactive_trajectory_with_velocities(data)
    traj_with_acc_json = generate_interactive_trajectory_with_accelerations(data)
    
    return render_template('report.html', 
                         data=data, 
                         formulas=formulas,
                         traj_json=traj_json, 
                         vel_json=vel_json, 
                         acc_json=acc_json,
                         traj_with_vel_json=traj_with_vel_json,
                         traj_with_acc_json=traj_with_acc_json)


@app.route('/export/pdf')
def export_pdf():
    pdf_data, error = generate_export_pdf(landscape=False)
    if error:
        return error, 500
    return send_file(io.BytesIO(pdf_data), as_attachment=True,
                     download_name='report.pdf', mimetype='application/pdf')


@app.route('/export/pdf16_9')
def export_pdf_16_9():
    pdf_data, error = generate_export_pdf(landscape=True)
    if error:
        return error, 500
    return send_file(io.BytesIO(pdf_data), as_attachment=True,
                     download_name='report_16_9.pdf', mimetype='application/pdf')


@app.route('/export/word')
def export_word():
    docx_data, error = generate_export_word(landscape=False)
    if error:
        return error, 500
    return send_file(io.BytesIO(docx_data), as_attachment=True,
                     download_name='report.docx', 
                     mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document')


@app.route('/export/word16_9')
def export_word_16_9():
    docx_data, error = generate_export_word(landscape=True)
    if error:
        return error, 500
    return send_file(io.BytesIO(docx_data), as_attachment=True,
                     download_name='report_16_9.docx', 
                     mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
