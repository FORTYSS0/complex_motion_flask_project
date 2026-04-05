import matplotlib.pyplot as plt
import io
import base64

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
