import os
import tempfile
import pdfkit
import pypandoc
from flask import render_template
from calc import compute_complex_motion
from app_helpers import latex_to_png


# Настройка wkhtmltopdf
WKHTMLTOPDF_PATH = '/usr/bin/wkhtmltopdf'
WKHTMLTOPDF_PATH = 'C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe'
if os.path.exists(WKHTMLTOPDF_PATH):
    PDFKIT_CONFIG = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)
else:
    print(f"Предупреждение: wkhtmltopdf не найден по пути {WKHTMLTOPDF_PATH}")
    PDFKIT_CONFIG = pdfkit.configuration()


# Подготовка данных для экспорта
def prepare_export_data():
    """Возвращает данные, формулы, изображения формул и пути к PNG для экспорта."""
    data, formulas = compute_complex_motion(t=1)
    
    # Преобразование формул в PNG
    formula_images = {key: latex_to_png(latex) for key, latex in formulas.items()}
    
    # Дополнительные формулы
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
    
    # Пути к статическим изображениям
    static_abs_path = os.path.abspath('static')
    img_files = {
        'trajectory': os.path.join(static_abs_path, 'trajectory.png'),
        'velocities': os.path.join(static_abs_path, 'velocities.png'),
        'accelerations': os.path.join(static_abs_path, 'accelerations.png'),
        'trajectory_with_velocities': os.path.join(static_abs_path, 'trajectory_with_velocities.png'),
        'trajectory_with_accelerations': os.path.join(static_abs_path, 'trajectory_with_accelerations.png'),
    }
    
    return data, formulas, formula_images, img_files


# Генерация PDF
def generate_export_pdf(landscape=False):
    """Генерирует PDF и возвращает (pdf_data, error)."""
    data, formulas, formula_images, img_files = prepare_export_data()
    
    rendered = render_template('report_export.html', 
                             data=data, 
                             formulas=formulas,
                             formula_images=formula_images, 
                             images=img_files,
                             export=True, 
                             landscape=landscape)
    
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
        f.write(rendered.encode('utf-8'))
        html_path = f.name
    
    pdf_path = html_path.replace('.html', '.pdf')
    options = {'enable-local-file-access': None}
    if landscape:
        options['orientation'] = 'Landscape'
    
    try:
        pdfkit.from_file(html_path, pdf_path, options=options, configuration=PDFKIT_CONFIG)
    except Exception as e:
        os.unlink(html_path)
        return None, f"Ошибка генерации PDF: {e}"
    
    with open(pdf_path, 'rb') as f:
        pdf_data = f.read()
    
    os.unlink(html_path)
    os.unlink(pdf_path)
    return pdf_data, None


# Генерация Word
def generate_export_word(landscape=False):
    """Генерирует Word (docx) и возвращает (docx_data, error)."""
    data, formulas, formula_images, img_files = prepare_export_data()
    
    rendered = render_template('report_export.html', 
                             data=data, 
                             formulas=formulas,
                             formula_images=formula_images, 
                             images=img_files,
                             export=True, 
                             landscape=landscape)
    
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
        return None, f"Ошибка генерации Word: {e}"
    
    os.unlink(html_path)
    os.unlink(docx_path)
    return docx_data, None