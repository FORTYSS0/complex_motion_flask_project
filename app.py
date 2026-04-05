from flask import Flask, render_template, send_file, url_for
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dsk.calc import dsk_compute_complex_motion
from dsk.plots import dsk_trajectory, \
    dsk_velocities, \
    dsk_accelerations

from sdt.calc import sdt_compute_complex_motion
from sdt.plots import sdt_trajectory, \
                sdt_velocities, \
                sdt_accelerations, \
                sdt_trajectory_with_velocities, \
                sdt_trajectory_with_accelerations

# from export_utils import generate_export_pdf, generate_export_word

app = Flask(__name__)
os.makedirs('static', exist_ok=True)


# Маршруты
@app.route('/')
def index1():
    data, formulas = dsk_compute_complex_motion(t=1)
    
    # Генерируем JSON для интерактивных графиков
    traj_json = dsk_trajectory(data)
    vel_json = dsk_velocities(data)
    acc_json = dsk_accelerations(data)
    
    return render_template('report_1_dsk.html', 
                         data=data, 
                         formulas=formulas,
                         traj_json=traj_json, 
                         vel_json=vel_json, 
                         acc_json=acc_json
    )


"""@app.route('/psk')
def index():
    data, formulas = compute_complex_motion(t=1)
    
    # Генерируем JSON для интерактивных графиков
    traj_json = generate_interactive_trajectory(data)
    vel_json = generate_interactive_velocities(data)
    acc_json = generate_interactive_accelerations(data)
    traj_with_vel_json = generate_interactive_trajectory_with_velocities(data)
    traj_with_acc_json = generate_interactive_trajectory_with_accelerations(data)
    
    return render_template('report_2_psk.html', 
                         data=data, 
                         formulas=formulas,
                         traj_json=traj_json, 
                         vel_json=vel_json, 
                         acc_json=acc_json,
                         traj_with_vel_json=traj_with_vel_json,
                         traj_with_acc_json=traj_with_acc_json
    )"""

"""@app.route('/eszd')
def index():
    data, formulas = compute_complex_motion(t=1)
    
    # Генерируем JSON для интерактивных графиков
    traj_json = generate_interactive_trajectory(data)
    vel_json = generate_interactive_velocities(data)
    acc_json = generate_interactive_accelerations(data)
    traj_with_vel_json = generate_interactive_trajectory_with_velocities(data)
    traj_with_acc_json = generate_interactive_trajectory_with_accelerations(data)
    
    return render_template('report_3_eszd.html', 
                         data=data, 
                         formulas=formulas,
                         traj_json=traj_json, 
                         vel_json=vel_json, 
                         acc_json=acc_json,
                         traj_with_vel_json=traj_with_vel_json,
                         traj_with_acc_json=traj_with_acc_json
    )"""


@app.route('/sdt')
def index4():
    data, formulas = sdt_compute_complex_motion(t=1)
    
    # Генерируем JSON для интерактивных графиков
    traj_json = sdt_trajectory(data)
    vel_json = sdt_velocities(data)
    acc_json = sdt_accelerations(data)
    traj_with_vel_json = sdt_trajectory_with_velocities(data)
    traj_with_acc_json = sdt_trajectory_with_accelerations(data)
    
    return render_template('report_4_sdt.html', 
                         data=data, 
                         formulas=formulas,
                         traj_json=traj_json, 
                         vel_json=vel_json, 
                         acc_json=acc_json,
                         traj_with_vel_json=traj_with_vel_json,
                         traj_with_acc_json=traj_with_acc_json
    )


"""@app.route('/export/pdf')
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
                     mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document')"""


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
