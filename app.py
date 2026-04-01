from flask import Flask, render_template, send_file, make_response
import os
import io
import tempfile
import pdfkit
import pypandoc
from calc import compute_complex_motion
from plots import generate_all_plots

app = Flask(__name__)

# Создаём папку для статики, если её нет
os.makedirs('static', exist_ok=True)

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

    # Подготавливаем HTML для экспорта с абсолютными путями к изображениям
    # Получаем абсолютные пути к графикам
    static_abs_path = os.path.abspath('static')
    img_files = {
        'trajectory': os.path.join(static_abs_path, 'trajectory.png'),
        'velocities': os.path.join(static_abs_path, 'velocities.png'),
        'accelerations': os.path.join(static_abs_path, 'accelerations.png'),
    }
    # Рендерим шаблон для экспорта
    rendered = render_template('report_export.html', data=data, formulas=formulas,
                               images=img_files, export=True)

    # Сохраняем HTML во временный файл
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
        f.write(rendered.encode('utf-8'))
        html_path = f.name

    # Конвертируем в PDF
    pdf_path = html_path.replace('.html', '.pdf')
    try:
        pdfkit.from_file(html_path, pdf_path, options={'enable-local-file-access': None})
    except Exception as e:
        os.unlink(html_path)
        return f"Ошибка генерации PDF: {e}", 500

    # Отправляем файл
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

    static_abs_path = os.path.abspath('static')
    img_files = {
        'trajectory': os.path.join(static_abs_path, 'trajectory.png'),
        'velocities': os.path.join(static_abs_path, 'velocities.png'),
        'accelerations': os.path.join(static_abs_path, 'accelerations.png'),
    }
    rendered = render_template('report_export.html', data=data, formulas=formulas,
                               images=img_files, export=True)

    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
        f.write(rendered.encode('utf-8'))
        html_path = f.name

    try:
        docx_data = pypandoc.convert_file(html_path, 'docx', outputfile=None)
    except Exception as e:
        os.unlink(html_path)
        return f"Ошибка генерации Word: {e}", 500

    os.unlink(html_path)

    return send_file(
        io.BytesIO(docx_data),
        as_attachment=True,
        download_name='report.docx',
        mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
