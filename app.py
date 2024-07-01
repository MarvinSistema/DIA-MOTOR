from flask import Flask, redirect, url_for
from DIA import asignacionDIA
import os

app = Flask(__name__)

# Registrar los Blueprints con sus prefijos de URL
app.register_blueprint(asignacionDIA, url_prefix='/asignacionDIA')

@app.route('/')
def index():
    return redirect(url_for('asignacionDIA.index'))  # Redirigir a la página principal

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
    