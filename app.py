from flask import Flask, render_template
from calc import compute_complex_motion
from plots import generate_all_plots
import os

app = Flask(__name__)

# Создаём папку для статики, если её нет
os.makedirs('static', exist_ok=True)

@app.route('/')
def index():
    data, formulas = compute_complex_motion(t=1)
    generate_all_plots(data)
    return render_template('report.html', data=data, formulas=formulas)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
