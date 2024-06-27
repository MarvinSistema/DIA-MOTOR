from flask import render_template, Blueprint
from datetime import datetime
from db_manager import fetch_data

planasEnPatio = Blueprint('planasEnPatio', __name__)
@planasEnPatio.route('/')
def index():
    planas, Operadores = cargar_datos()
    planas = procesar_planas(planas)
    operadores_sin_asignacion = procesar_operadores(Operadores)
    html_empates_dobles = planas.to_html()
    html_operadores = operadores_sin_asignacion.to_html()
    return render_template('planasEnPatio.html', datos_html= html_empates_dobles, datos_html_operadores=html_operadores)

def cargar_datos():
    consulta_planas = """
        SELECT *
        FROM DimTableroControlRemolque
        WHERE PosicionActual = 'NYC'
        AND Estatus = 'CARGADO EN PATIO'
        AND Ruta IS NOT NULL
        AND CiudadDestino != 'MONTERREY'
        AND CiudadDestino != 'GUADALUPE'
        AND CiudadDestino != 'APODACA'
    """
    consulta_operadores = """
        SELECT * 
        FROM DimTableroControl
        """
    planas = fetch_data(consulta_planas)
    Operadores = fetch_data(consulta_operadores)  
    return planas, Operadores

def procesar_planas(planas):
    planas['Horas en patio'] = ((datetime.now() - planas['FechaEstatus']).dt.total_seconds() / 3600.0).round(1)
    #planas['FechaEstatus'] = planas['FechaEstatus'].dt.strftime('%Y-%m-%d %H:%M')
    planas['ValorViaje'] = planas['ValorViaje'].apply(lambda x: "${:,.0f}".format(x))
    planas.sort_values(by=['FechaEstatus'], ascending=True, inplace=True)
    planas = planas[['Remolque', 'CiudadDestino', 'ValorViaje', 'Horas en patio']]
    planas.reset_index(drop=True, inplace=True)
    planas.index += 1
    return planas

def procesar_operadores(Operadores):
    Operadores = Operadores[(Operadores['Estatus'] == 'Disponible') & (Operadores['Destino'] == 'NYC')]
    Operadores  = Operadores [Operadores ['UOperativa'].isin(['U.O. 01 ACERO', 'U.O. 02 ACERO', 'U.O. 03 ACERO', 'U.O. 04 ACERO', 'U.O. 07 ACERO','U.O. 39 ACERO', 'U.O. 15 ACERO (ENCORTINADOS)', 'U.O. 41 ACERO LOCAL (BIG COIL)', 'U.O. 52 ACERO (ENCORTINADOS SCANIA)'])]
    Operadores['Tiempo Disponible'] = ((datetime.now() - Operadores['FechaEstatus']).dt.total_seconds() / 3600).round(1)
    Operadores = Operadores[Operadores['ObservOperaciones'].isna() | Operadores['ObservOperaciones'].eq('')]
    Operadores = Operadores[['Operador', 'Tractor', 'UOperativa', 'Tiempo Disponible']]
    Operadores.sort_values(by='Tiempo Disponible', ascending=False, inplace=True)
    Operadores.reset_index(drop=True, inplace=True)
    Operadores.index += 1 
    return Operadores