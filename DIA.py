import os
from datetime import datetime, timedelta

import networkx as nx
import numpy as np
import pandas as pd
import requests
import urllib3
from flask import Blueprint, render_template
from sklearn.metrics import DistanceMetric
import gdown
import io
from concurrent.futures import ThreadPoolExecutor
from db_manager import fetch_data, fetch_data_PRO, fetch_data_DIA, create_engine_db_DIA

asignacionDIA = Blueprint('asignacionDIA', __name__)
@asignacionDIA.route('/')
def index():
    global asignacion 
    asignacion  = None
    planas, Operadores, Cartas, Gasto, Km, Bloqueo, ETAs, Permisos, DataDIA, MttoPrev, CheckTaller, OrAbierta, Op, Tractor, Dolly  = cargar_datos()
    planasSAC = planas_sac()
    emparejamientosPla= emparejamientosPlanas(planas, DataDIA, planasSAC)
    operadores_sin_asignacion = procesar_operadores(Operadores, DataDIA, Dolly)
    asignacionesPasadasOperadores=  asignacionesPasadasOp(Cartas)
    siniestroKm= siniestralidad(Gasto, Km)
    ETAi= eta(ETAs)
    PermisosOp= permisosOperador(Permisos)
    cerca = cercaU()
    calOperadores= calOperador(operadores_sin_asignacion, Bloqueo, asignacionesPasadasOperadores, siniestroKm, ETAi, PermisosOp, cerca, MttoPrev, CheckTaller, OrAbierta )
    asignacion = asignacion2(emparejamientosPla, calOperadores, planas, Op, Tractor)
    #conexionSPL= api_spl()
    #datosAzure=insertar_datos()
    #borrarDataAzure=borrar_datos_antiguos()
    datos_html = asignacion.to_html()
    
    return render_template('asignacionDIA.html', datos_html=datos_html)

def cargar_datos():
    consultas = [
        ("fetch_data", """
            SELECT *
            FROM DimTableroControlRemolque_CPatio
            WHERE PosicionActual = 'NYC'
            AND Estatus = 'CARGADO EN PATIO'
            AND Ruta IS NOT NULL
            AND CiudadDestino NOT IN ('MONTERREY', 'GUADALUPE', 'APODACA')
            AND Remolque != 'P3169'
        """),
        ("fetch_data", "SELECT * FROM DimTableroControl_Disponibles"),
        ("fetch_data", """
            SELECT IdViaje, FechaSalida, Operador, Tractor, UnidadOperativa, Cliente, SubtotalMXN, Ruta, IdConvenio 
            FROM ReporteCartasPorte 
            WHERE FechaSalida > DATEADD(day, -90, GETDATE())
        """),
        ("fetch_data_PRO", """
            SELECT Reporte, Empresa, Tractor, FechaSiniestro, TotalFinal
            FROM DimReporteUnificado
            WHERE FechaSiniestro > DATEADD(day, -90, GETDATE())
        """),
        ("fetch_data", """
            SELECT NombreOperador, FechaPago, Tractor, KmsReseteo  
            FROM DimRentabilidadLiquidacion 
            WHERE FechaPago > DATEADD(day, -90, GETDATE())
        """),
        ("fetch_data", """
            SELECT NombreOperador, Activo, OperadorBloqueado
            FROM DimOperadores
            WHERE Activo = 'Si'
        """),
        ("fetch_data", """
            SELECT NombreOperador, FechaFinalizacion, CumpleETA 
            FROM DimIndicadoresOperaciones 
            WHERE FechaLlegada > DATEADD(day, -90, GETDATE()) 
            AND FechaLlegada IS NOT NULL
        """),
        ("fetch_data", """
            SELECT NoOperador, Nombre, Activo, FechaBloqueo
            FROM DimBloqueosTrafico
        """),
        ("fetch_data_DIA", "SELECT * FROM DIA_NYC"),
        ("fetch_data", """
            SELECT ClaveTractor, UltimoMantto, Descripcion, VencimientoD
            FROM DimPreventivoFlotillas
            WHERE VencimientoD>0.97
        """),
        ("fetch_data", """
            SELECT Tractor, UnidadOperativa, Estatus, FechaEstatus
            FROM (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY Tractor ORDER BY FechaEstatus DESC) as rn
                FROM DimDashboardHistorico
                WHERE Tractor != ''
            ) t
            WHERE rn <= 5
            AND (Estatus = 'Check Express' OR Estatus = 'En Taller')
            ORDER BY Tractor ASC, FechaEstatus DESC
        """),
        ("fetch_data", """
            SELECT IdDimOrdenReparacion, IdOR, TipoEquipo, ClaveEquipo, FechaCreacion, FechaFinalizacion
            FROM DimOrdenesReparacion
            WHERE FechaFinalizacion IS NULL AND TipoEquipo = 'Tractor' AND Descripcion != 'Orden de Reparacion Especial'
        """),
        ("fetch_data", """
            SELECT IdOperador,NombreOperador, Activo, OperadorBloqueado
            FROM DimOperadores
            WHERE Activo = 'Si'
        """),
        ("fetch_data", """
            SElECT * 
            FROM Cat_Tractor 
            Where Activo = 'Si'
        """),
        ("fetch_data", """
            SELECT * 
            FROM 
            Cat_Dolly
        """)
    ]
    
    
    with ThreadPoolExecutor() as executor:
        # Lanzar cada consulta en un hilo separado
        futures = [executor.submit(globals()[func], sql) for func, sql in consultas]
        # Esperar a que todas las consultas se completen y recoger los resultados
        results = [future.result() for future in futures]

    return tuple(results)

def procesar_operadores(Operadores, DataDIA, Dolly):
    u_operativas = ['U.O. 01 ACERO', 'U.O. 02 ACERO', 'U.O. 03 ACERO', 'U.O. 04 ACERO', 'U.O. 06 ACERO (TENIGAL)',  'U.O. 07 ACERO', 'U.O. 39 ACERO']
    DollyDisponibles = Dolly.copy()
    DollyDisponibles = DollyDisponibles[
        (DollyDisponibles['Activo'] == True) &
        (DollyDisponibles['UbicacionActual'] == 'NYC') &
        (DollyDisponibles['EstatusActual'] == 'Disponible') &
        (DollyDisponibles['Unidad'] != 'U.O. 100 - EQUIPO NUEVO') &
        (DollyDisponibles['Unidad'] != 'U.O. 14 ACERO (PATIOS MX)')
    ]

    DollyDisponibles = DollyDisponibles[['IdDolly', 'ClaveDolly']]
        
    Operadores= pd.merge(Operadores, Dolly, left_on='DollyAsignado', right_on='ClaveDolly', how='left')
    Operadores = Operadores[(Operadores['Estatus'] == 'Disponible') & (Operadores['Destino'] == 'NYC')]
    Operadores  = Operadores [Operadores ['UOperativa'].isin(['U.O. 01 ACERO', 'U.O. 02 ACERO', 'U.O. 03 ACERO', 'U.O. 04 ACERO', 'U.O. 06 ACERO (TENIGAL)', 'U.O. 07 ACERO','U.O. 39 ACERO'])]
    Operadores['Tiempo Disponible'] = ((datetime.now() - Operadores['FechaEstatus']).dt.total_seconds() / 3600).round(1)
    Operadores = Operadores[Operadores['Tiempo Disponible'] > 0.15]
    Operadores= Operadores[~Operadores['Operador'].isin(DataDIA['Operador'])]#Elimina Ops ya asignados

    
    #Verificamos que los dollys no esten asignados ya a tractores (operadores)
    DollyDisponibles['IdDolly'] = DollyDisponibles['IdDolly'].astype(int)
    Operadores['IdDolly'] = Operadores['IdDolly'].dropna().astype(int)
    DollyDisponibles= DollyDisponibles[~DollyDisponibles['IdDolly'].isin(Operadores['IdDolly'])]
    # Encuentra índices donde IdDolly es NaN y la unidad operativa está en la lista deseada
    mask = (pd.isna(Operadores['IdDolly'])) & (Operadores['UOperativa'].isin(u_operativas))
    # Genera una lista de posibles IdDolly para asignar desde DollyDisponibles
    dolly_ids = DollyDisponibles['IdDolly'].unique()
    # Asignar aleatoriamente un IdDolly desde DollyDisponibles a los índices filtrados en Operadores
    Operadores.loc[mask, 'IdDolly'] = np.random.choice(dolly_ids, size=mask.sum())
    
    
    Operadores = Operadores[['Operador', 'Tractor', 'UOperativa', 'Tiempo Disponible', 'DollyAsignado', 'ClaveDolly', 'IdDolly']]
    Operadores.sort_values(by='Tiempo Disponible', ascending=False, inplace=True)
    Operadores.reset_index(drop=True, inplace=True)
    Operadores.index += 1
    return Operadores

def calOperador(operadores_sin_asignacion, Bloqueo, asignacionesPasadasOp, siniestroKm, ETAi, PermisosOp, cerca, MttoPrev, CheckTaller, OrAbierta) :
    calOperador= operadores_sin_asignacion.copy()
    calOperador= pd.merge(operadores_sin_asignacion, Bloqueo, left_on='Operador', right_on='NombreOperador', how='left')
    calOperador= calOperador[calOperador['Tractor'].isin(cerca['cve_uni'])]
    calOperador= pd.merge(calOperador, asignacionesPasadasOp, left_on='Operador', right_on='Operador', how='left')
    calOperador= pd.merge(calOperador, siniestroKm, left_on='Tractor', right_on='Tractor', how='left')
    calOperador= pd.merge(calOperador, PermisosOp, left_on='Operador', right_on='Nombre', how='left')
    calOperador= pd.merge(calOperador, ETAi, left_on='Operador', right_on='NombreOperador', how='left')
    calOperador['OrAbierta'] = calOperador['Tractor'].apply(
    lambda x: 'Si' if x in OrAbierta['ClaveEquipo'].values else 'No'
    )
    calOperador['Pasar a Mtto Preventivo'] = calOperador['Tractor'].apply(
    lambda x: 'Si' if x in MttoPrev['ClaveTractor'].values else 'No'
    )
    calOperador['¿Paso por Check/Mtto?'] = calOperador['Tractor'].apply(
    lambda x: 'Si' if x in CheckTaller['Tractor'].values else 'No'
    )
    
    calOperador= calOperador[calOperador['OrAbierta'] == 'No']
    calOperador= calOperador[calOperador['Pasar a Mtto Preventivo'] == 'No']
    calOperador= calOperador[calOperador['¿Paso por Check/Mtto?'] == 'Si']
    
    
    #Control Valores FAltantes    
    if 'Calificacion SAC' not in calOperador.columns:
        calOperador['Calificacion SAC'] = 0  
    calOperador['ViajeCancelado']= 20
    
    
    # Generar números aleatorios entre 25 y 50
    random_values = np.random.randint(25, 51, size=len(calOperador))
    # Convertir el ndarray en una serie de pandas
    random_series = pd.Series(random_values, index=calOperador.index)
    # Reemplazar los valores nulos con los valores aleatorios generados
    calOperador['CalificacionVianjesAnteiores'] = calOperador['CalificacionVianjesAnteiores'].fillna(random_series)
    calOperador['PuntosSiniestros'] = calOperador['PuntosSiniestros'].fillna(20)
    calOperador['Calificacion SAC'] = calOperador['Calificacion SAC'].fillna(20)
    calOperador['Activo_y'] = calOperador['Activo_y'].fillna('No')

    #calOperador['CalFinal']= calOperador['CalificacionVianjesAnteiores']+calOperador['PuntosSiniestros']+calOperador['Calificacion SAC']+calOperador['ViajeCancelado']
    calOperador['CalFinal'] = (
    calOperador['CalificacionVianjesAnteiores'] +
    calOperador['PuntosSiniestros'] +
    calOperador['Calificacion SAC'] +
    calOperador['ViajeCancelado'] +
    (calOperador['Tiempo Disponible'] * 0.2)
    )
    calOperador = calOperador[['Operador', 'Activo_y','Tractor', 'UOperativa', 'Tiempo Disponible', 'OperadorBloqueado', 
        'Bueno','Regular', 'Malo', 'CalificacionVianjesAnteiores', 'Siniestralidad', 'PuntosSiniestros', 'Cumple ETA', 'No Cumple ETA',
        'Calificacion SAC', 'ViajeCancelado', 'CalFinal', 'IdDolly']]
   
   
    calOperador = calOperador.rename(columns={
    'OperadorBloqueado': 'Bloqueado Por Seguridad',
    'CalificacionVianjesAnteiores': 'Calificacion ViajesAnteiores',
    'PuntosSiniestros': 'Puntos Siniestros',
    'ViajeCancelado': 'Viaje Cancelado',
    'Activo_y': 'Permiso'
    })

    
    
    calOperador.rename(columns={
        'Viaje Cancelado': 'Actitud'
        }, inplace=True)
    
    return calOperador

def asignacionesPasadasOp(Cartas):
    CP= Cartas.copy()
    # 30 dias atras
    fecha_actual = datetime.now()
    fecha_30_dias_atras = fecha_actual - timedelta(days=80)
    # Dividir la columna 'Ruta' por '||' y luego por '-' para obtener origen
    CP[['ID1', 'Ciudad_Origen', 'Ciudad_Destino']] = CP['Ruta'].str.split(r' \|\| | - ', expand=True)
    # Filtro Mes actual, UO, ColumnasConservadas, Ciudad Origen
    CP = CP[CP['FechaSalida'] >= fecha_30_dias_atras]
    CP = CP[CP['UnidadOperativa'].isin(['U.O. 01 ACERO', 'U.O. 02 ACERO', 'U.O. 03 ACERO', 'U.O. 04 ACERO','U.O. 06 ACERO (TENIGAL)', 'U.O. 39 ACERO', 'U.O. 07 ACERO', 'U.O. 15 ACERO (ENCORTINADOS)', 'U.O. 52 ACERO (ENCORTINADOS SCANIA)',  'U.O. 41 ACERO LOCAL (BIG COIL)'])]
    CP= CP[['IdViaje', 'Cliente', 'Operador', 'Ruta', 'SubtotalMXN', 'FechaSalida', 'Ciudad_Origen']]
    CP = CP[CP['Ciudad_Origen'].isin(['MONTERREY'])]

    # Agrupar 
    CP = CP.groupby(['IdViaje', 'Operador']).agg({'SubtotalMXN': 'sum'}).reset_index()

    # Funsion para determinar tipo de viaje
    def etiquetar_tipo_viaje(subtotal):
        if subtotal >= 105000:
            return "Bueno"
        elif subtotal <= 81000:
            return "Malo"
        else:
            return "Regular"

    # Aplicar la función a la columna 'SubtotalMXN' para crear la columna 'TipoViaje'
    CP['TipoViaje'] = CP['SubtotalMXN'].apply(lambda subtotal: etiquetar_tipo_viaje(subtotal))

    # Crear una tabla pivote para contar la cantidad de 'Bueno', 'Malo' y 'Regular' para cada operador
    CP = pd.pivot_table(CP, index='Operador', columns='TipoViaje', aggfunc='size', fill_value=0)

    # Calcula el puntaje bruto para cada fila
    CP['PuntajeBruto'] = (CP['Malo'] * 0.5)+ (CP['Regular'] * 1) + (CP['Bueno'] * 2)

    # Calcula el máximo y mínimo puntaje bruto que podría existir basado en los datos del DataFrame
    min_puntaje_posible = CP['PuntajeBruto'].min()
    max_value = CP['PuntajeBruto'].max()

    #Normalizo entre 1-50 el puntaje buruto
    CP['CalificacionVianjesAnteiores'] =  50-(1 + (49 * (CP['PuntajeBruto'] - min_puntaje_posible) / (max_value - min_puntaje_posible)))
    CP['CalificacionVianjesAnteiores'] = CP['CalificacionVianjesAnteiores'].replace([float('inf'), -float('inf')], 0)
    CP['CalificacionVianjesAnteiores'] = CP['CalificacionVianjesAnteiores'].round().astype(int)
    CP = CP.reset_index()
    return CP

def siniestralidad(Gasto, Km):
    G= Gasto.copy()
    K = Km.copy()

    # Filtrar las filas que contengan "SINIESTRO" en la columna "Reporte", la empresa NYC
    G = G[G['Reporte'].str.contains("SINIESTRO")]
    G= G[G['Empresa'].str.contains("NYC")]

    # Quedarme con tres meses de historia hacia atras a partir de hoy, mantener las columnas
    G= G[pd.to_datetime(G['FechaSiniestro']).dt.date >= (datetime.now() - timedelta(days=3*30)).date()]
    G= G[["Tractor","TotalFinal"]]

    # Agrupar Por Tractor
    G = G.groupby('Tractor')['TotalFinal'].sum()

    # Resetear Index
    G= G.reset_index()

    # Quedarse con columnas
    K = K[["Tractor", "FechaPago", "KmsReseteo"]]

    # Quedarme con tres meses de historia hacia atras a partir de hoy
    K = K[pd.to_datetime(K['FechaPago']).dt.date >= (datetime.now() - timedelta(days=3*30)).date()]

    # Agrupar por la columna "Tractor" y sumar los valores de la columna "KmsReseteo"
    K= K.groupby('Tractor')['KmsReseteo'].sum()

    # Voy a pasarlo a un dataframe 
    K = K.reset_index()

    # Realizar left join entre kilometros y gasto
    K= K.merge(G, on='Tractor', how='left')

    # Rellenar los valores NaN en la columna "totalfinal" con ceros (0)
    K['TotalFinal'] = K['TotalFinal'].fillna(0)

    # Agregar una nueva columna llamada "SINIESTRALIDAD" al DataFrame resultado_join
    K['Siniestralidad'] = (K['TotalFinal'] / K['KmsReseteo']).round(2)

    # Ordenar el DataFrame resultado_join de menor a mayor por la columna "SINIESTRALIDAD"
    K= K.sort_values(by='Siniestralidad')

    # Funsion para asiganar puntaje
    def Sis(SiniestroP):
        if SiniestroP >= 0.15:
            return 0
        elif SiniestroP >= 0.06:
            return 10
        else:
            return 20
        
    # Asignar los puntajes por fila
    K['PuntosSiniestros'] = K['Siniestralidad'].apply(lambda SiniestroP: Sis(SiniestroP))

    #voy a pasarlo a un dataframe 
    K = K.reset_index()
    return K

def permisosOperador(Permisos):
    Permisos= Permisos.sort_values(by=['Nombre', 'FechaBloqueo'], ascending=[False, False])
    Permisos= Permisos.drop_duplicates(subset='Nombre', keep='first')
    Permisos = Permisos.query('Activo == "Si" & ~(Activo == "No")')
    return Permisos

def eta(ETAi):
    # Intentar crear una tabla pivote para contar la cantidad de 'Bueno', 'Malo' y 'Regular' para cada operador
    
    ETAi = ETAi.pivot_table(index='NombreOperador', columns='CumpleETA', aggfunc='size', fill_value=0).reset_index()

    # Asegurar que las columnas 'Cumple' y 'No Cumple' están presentes antes de realizar cálculos
    if 'Cumple' in ETAi.columns and 'No Cumple' in ETAi.columns:
        ETAi['Cumple'] = ETAi['Cumple'].fillna(0)
        ETAi['No Cumple'] = ETAi['No Cumple'].fillna(0)
        ETAi['Calificacion SAC'] = ((ETAi['Cumple'] / (ETAi['Cumple'] + ETAi['No Cumple'])) * 10).round(0).astype(int)
    else:
        print("Las columnas necesarias 'Cumple' o 'No Cumple' no están presentes.")
        return ETAi  # Retorna el DataFrame sin la columna 'Calificacion SAC'

    # Cambiar los nombres de las columnas para reflejar los datos correctamente
    ETAi.rename(columns={
        'Cumple': 'Cumple ETA',
        'No Cumple': 'No Cumple ETA'
    }, inplace=True)
    

    return ETAi

def asignacion2(planasPorAsignar, calOperador, planas, Op, Tractor):
    
    
    calOperador= calOperador[calOperador['Bloqueado Por Seguridad'].isin(['No'])]
    calOperador= calOperador[calOperador['Permiso'].isin(['No'])]
    
    calOperador = calOperador[calOperador['UOperativa'].isin(['U.O. 01 ACERO', 'U.O. 02 ACERO', 'U.O. 03 ACERO', 'U.O. 04 ACERO', 'U.O. 06 ACERO (TENIGAL)', 'U.O. 07 ACERO','U.O. 39 ACERO'])]
    calOperador = calOperador.sort_values(by=['CalFinal', 'Tiempo Disponible'], ascending=[False, False])
    calOperador = calOperador.reset_index(drop=True)
    calOperador.index = calOperador.index + 1
    cantidadOperadores = len(calOperador)
            

    planasPorAsignar= planasPorAsignar.head(cantidadOperadores)
    planasPorAsignar= planasPorAsignar.sort_values(by='Monto', ascending=False)
    planasPorAsignar.reset_index(drop=True, inplace=True)
    planasPorAsignar.index = planasPorAsignar.index + 1
    
    
    #f_concatenado = pd.concat([planasPorAsignar, calOperador], axis=1)
    f_concatenado= pd.merge(planasPorAsignar, calOperador, left_index=True, right_index=True, how='left')
    # Unión para Plana 1
    f_concatenado = pd.merge(f_concatenado, planas, left_on='remolque_a', right_on='Remolque', how='left', suffixes=('', '_right1'))
    f_concatenado.rename(columns={'IdSolicitud': 'IdSolicitud1', 'IdRemolque': 'IdRemolque1'}, inplace=True)

    # Unión para Plana 2
    f_concatenado= pd.merge(f_concatenado, planas, left_on='remolque_b', right_on='Remolque', how='left', suffixes=('', '_right2'))
    f_concatenado.rename(columns={'IdSolicitud': 'IdSolicitud2', 'IdRemolque': 'IdRemolque2'}, inplace=True)
    f_concatenado= pd.merge(f_concatenado, Op, left_on='Operador', right_on='NombreOperador', how='left')
    f_concatenado= pd.merge(f_concatenado, Tractor, left_on='Tractor', right_on='ClaveTractor', how='left')
    
    #f_concatenado['IdOperador'] = f_concatenado['IdOperador'].astype(int)
    f_concatenado = f_concatenado[['Ruta', 'Operador', 'Tractor', 'remolque_a', 'remolque_b', 'Tiempo Disponible', 'IdOperador', 'IdTractor', 'IdRemolque1', 'IdSolicitud1', 'IdRemolque2', 'IdSolicitud2', 'IdDolly']]
    
    
    f_concatenado.rename(columns={
    'remolque_a': 'Plana 1',
    'remolque_b': 'Plana 2',
    }, inplace=True)


    f_concatenado = f_concatenado[f_concatenado['Operador'].notna()]
    
    return f_concatenado

def api_spl():
    global asignacion 
    a = asignacion[['IdSolicitud1', 'IdSolicitud2', 'IdRemolque1', 'IdRemolque2', 'IdTractor', 'IdOperador','IdDolly']]
    

    def token_api():
        url = 'https://splpro.mx/ApiSpl/api/Login/authenticate'
        credentials = {
            'UserName': os.getenv('USERNAME'),
            'Password': os.getenv('PASSWORD'),
            'IdEmpresa': os.getenv('ID_EMPRESA', 1)  # El segundo argumento es el valor por defecto
        }
        response = requests.post(url, json=credentials, verify=False)
        response_data = response.json()

        if response.status_code == 200 and response_data.get('Success'):
            token = response_data.get('Token')
            print(f'Token recibido: {token}')
            return token
        else:
            print(f'Error en la solicitud: {response.status_code}')
            print(f'Mensaje del error: {response_data.get("Message")}')
            return None

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    token = token_api()
    if token:
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        url = 'https://splpro.mx/ApiSpl/api/ArmadoFull/UnirSolicitudes'
        
        
        a = a.fillna(0)  # Rellena los NaN con 0
        
        # Itera sobre cada fila del DataFrame y envía una solicitud por cada fila
        for index, row in a.iterrows():
            payload = {
                'IdSolicitud1': int(row['IdSolicitud1']),
                'IdSolicitud2': int(row['IdSolicitud2']),
                'IdRemolque1': int(row['IdRemolque1']),
                'IdRemolque2': int(row['IdRemolque2']),
                'IdTractor': int(row['IdTractor']),
                'IdOperador': int(row['IdOperador']),
                'IdDolly':int(row['IdDolly'])
            }

            response = requests.post(url, json=payload, headers=headers, verify=False)
            if response.status_code == 200:
                print(f"Solicitud procesada correctamente para el índice {index}")
                print(response.json())
            else:
                print(f"Error en la solicitud para el índice {index}: {response.status_code}")
                print(response.text)

        # Suprime advertencias de SSL (opcional)
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    return a

def cercaU():
    # Paso 1: Login y obtención del token
    url_login = 'http://74.208.129.205:3000/loginUser'
    payload_login = {
        'Usuario':  os.getenv('USUARIOTRACK'),
        'Password': os.getenv('PASSWORDTRACK')
    }
    
    
    # Intento de login
    response_login = requests.post(url_login, json=payload_login)
    if response_login.status_code != 200:
        print("Error en la conexión para login:", response_login.status_code)
        print("Detalle del error:", response_login.text)
        return None

    # Extracción del token
    token = response_login.json().get('token')
    print("Conexión exitosa para login! Token obtenido.")

    # Paso 2: Obtención de datos usando el token
    url_datos = 'http://74.208.129.205:3000/clientes/GNYC/TableroDeControlSPL'
    headers_datos = {'Authorization': f'Bearer {token}'}
    body_datos = {'idEmpresa': 1}
    
    # Solicitud de datos
    response_datos = requests.post(url_datos, headers=headers_datos, json=body_datos)
    if response_datos.status_code != 200:
        print("Error en la conexión para obtener datos:", response_datos.status_code)
        print("Detalle del error:", response_datos.text)
        return None

    # Conversión de los datos a DataFrame
    datos_empresa = response_datos.json()
    cerca= pd.DataFrame(datos_empresa)
    print("Conexión exitosa para obtener datos de la empresa!")
    cerca = cerca.loc[cerca['localizacion'] == '0.00 Km. NYC MONTERREY']
    cerca= cerca[['cve_uni']]
    return cerca

def insertar_datos():
    global asignacion
    if not isinstance(asignacion, pd.DataFrame):
        print("asignacion no está definido correctamente como un DataFrame.")
        return  # Salir de la función si asignacion no es un DataFrame
    
    conn = create_engine_db_DIA()
    if conn is not None:
        try:
            cursor = conn.cursor()
            # Añadir la columna de fecha en el query de inserción
            query = "INSERT INTO DIA_NYC (Operador, Plana, FechaCreacion) VALUES (?, ?, ?)"
            for index, row in asignacion.iterrows():
                # Iterar sobre cada 'Plana' y crear un registro separado
                for plana in ['Plana 1', 'Plana 2']:
                    if not pd.isnull(row[plana]) and row[plana] != 0:
                        # Obtener la fecha y hora actual
                        fecha_actual = datetime.now()
                        # Ejecutar la consulta con la fecha incluida
                        cursor.execute(query, (row['Operador'], row[plana], fecha_actual))
            conn.commit()
            print("Datos insertados correctamente.")
        except Exception as e:
            print(f"Error al insertar los datos: {e}")
        finally:
            cursor.close()
            conn.close()
    else:
        print("No se pudo establecer la conexión con la base de datos.")
        
def borrar_datos_antiguos():
    conn = create_engine_db_DIA()  # Asume que esta función retorna una conexión activa
    if conn is not None:
        try:
            cursor = conn.cursor()
            # Asume que tienes una columna 'FechaCreacion' en la tabla 'DIA_NYC'
            query = "DELETE FROM DIA_NYC WHERE FechaCreacion <= DATEADD(hour, -7, GETDATE())" #zona horaria en azure incorrecta
            cursor.execute(query)
            conn.commit()  # Confirma la transacción
            print("Registros antiguos eliminados correctamente mayor a una hora")
        except Exception as e:
            print(f"Error al borrar datos: {e}")
        finally:
            cursor.close()
        conn.close()
    else:
        print("No se pudo establecer la conexión con la base de datos.")
        
def planas_sac():
    url = 'https://drive.google.com/uc?id=1h3oynOXp11tKAkNmq4SkjBR8q_ZyJa2b'
    path = gdown.download(url, output=None, quiet=False)  # Guarda el archivo temporalmente

    # Abrir el archivo temporal en modo binario
    with open(path, 'rb') as f:
        data = io.BytesIO(f.read())  # Leer los datos del archivo y pasarlos a BytesIO

    # Leer el archivo desde el buffer directamente en un DataFrame
    df = pd.read_excel(data)
    
    # Ordenar el DataFrame por la columna 'Cita de descarga' en orden descendente
    df = df.sort_values(by='fecha de salida', ascending=False, na_position='last')
    df = df.groupby('Remolque')['fecha de salida'].max().reset_index()
    
    return df

def emparejamientosPlanas(planas, DataDIA, planasSAC):
    
    planas= planas[~planas['Remolque'].isin(DataDIA['Plana'])]
    planas = pd.merge(planas, planasSAC, on='Remolque', how='left')
    planas['Horas en patio'] = ((datetime.now() - planas['fecha de salida']).dt.total_seconds() / 3600.0).round(1)
    planas = planas.sort_values(by='Horas en patio', ascending=False)
    planas = planas[['IdSolicitud', 'Remolque', 'CiudadDestino', 'Horas en patio', 'ValorViaje']]
    planas = planas[planas['CiudadDestino'] != 'ALLENDE'] 
    
    def emparejar_misma_ciudad_mayor23(planas):
        # Filtrar las planas con más de 23 horas en el patio
        df_filtrado = planas[planas['Horas en patio'] > 23]
        
        # Listas para almacenar las parejas formadas
        parejas = []
        
        # Agrupar por 'CiudadDestino'
        destinos_agrupados = df_filtrado.groupby('CiudadDestino')
        
        for destino, grupo in destinos_agrupados:
            # Si hay más de una plana en el mismo destino
            if len(grupo) > 1:
                ids = grupo['IdSolicitud'].tolist()
                remolques = grupo['Remolque'].tolist()
                horas = grupo['Horas en patio'].tolist()
                valores = grupo['ValorViaje'].tolist()
                for i in range(0, len(remolques) - 1, 2):
                    if i + 1 < len(remolques):
                        parejas.append({
                            'Destino': destino,
                            'IdSolicitud1': ids[i],
                            'IdSolicitud2': ids[i + 1],
                            'Remolque1': remolques[i],
                            'Remolque2': remolques[i + 1],
                            'Horas en patio1': horas[i],
                            'Horas en patio2': horas[i + 1],
                            'ValorViaje1': valores[i],
                            'ValorViaje2': valores[i + 1]
                        })
 
        
        # Convertir las parejas a un DataFrame para mostrar el resultado
        parejas_df = pd.DataFrame(parejas)
        
        
        if not parejas_df.empty:
            ids_emparejados = pd.concat([parejas_df['IdSolicitud1'], parejas_df['IdSolicitud2']])
            planas_restante = planas[~planas['IdSolicitud'].isin(ids_emparejados)]
        else:
            planas_restante = planas.copy()
        
        return parejas_df, planas_restante
    
    def emparejar_destinos_cercanos_mayor23(planas, distancia_maxima=200):
        
        _, data_restante = emparejar_misma_ciudad_mayor23(planas)
        
       
        Ubicaciones = pd.DataFrame({
            'City': ['ELPRIETO,VERACRUZ', 'GUADALUPEZACATECAS', 'TEZOYUCA', 'CUAUTLA', 'CDVALLES', 'ACAPULCO', 'XALAPA,VER','AMATLANDELOSREYES', 'CUAUTLA,MORELOS','QUERETARO', 'GUADALAJARA', 'PUERTOVALLARTA', 'MAZATLAN', 'CULIACAN', 'LEON', 'MEXICO', 'SANLUISPOTOSI', 'VERACRUZ', 'TULTITLAN', 'JIUTEPEC', 'VILLAHERMOSA', 'PACHUCADESOTO', 'COLON', 'MERIDA', 'SALTILLO', 'CHIHUAHUA', 'TUXTLAGTZ', 'CORDOBA',
                        'TOLUCA', 'CIUDADHIDALGOCHP', 'CAMPECHE', 'ATITALAQUIA', 'MATAMOROS', 'ZAPOPAN', 'CIUDADCUAHUTEMOCCHH', 'MORELIA', 'TLAXCALA', 'GUADALUPE', 'SANTACRUZSON', 'LASVARAS', 'PACHUCA', 'CIUDADJUAREZ', 'TLAJOMULCO', 'PIEDRASNEGRAS', 'RAMOSARIZPE', 'ORIZABA', 'TAPACHULA', 'TEPATITLAN', 'TLAQUEPAQUE', 'TEAPEPULCO', 'LABARCA', 'ELMARQUEZ', 'CIUDADVICTORIA', 'NUEVOLAREDO', 'TIZAYUCA,HIDALGO', 'ELSALTO', 'OCOTLANJAL', 'TEZONTEPEC', 'ZAPOTILTIC', 'PASEOELGRANDE', 'POZARICA', 'JACONA', 'FRESNILLO', 'PUEBLA', 'TUXTLAGUTIERREZ', 'PLAYADELCARMEN', 'REYNOSA', 'MEXICALI', 'TEPEJIDELORODEOCAMPO',
                        'LEON', 'CUERNAVACA', 'CHETUMAL', 'CHIHUAHUA', 'SILAO', 'ACAPULCODEJUAREZ', 'AGUASCALIENTES', 'TIJUANA', 'OCOSINGO', 'MONCLOVA', 'OAXACA', 'SOLIDARIDAROO', 'JIUTEPEC', 'ELPRIETO', 'TORREON', 'HERMOSILLO', 'CELAYA', 'CANCUN', 'URUAPAN', 'ALTAMIRA', 'COATZACUALCOS', 'IRAPUATO', 'CASTAÑOS', 'DURANGO', 'COLON', 'CIUDADVALLLES', 'MANZANILLA', 'TAMPICO', 'GOMEZPALACIO', 'ZACATECAS', 'SALAMANCA', 'COMITANDEDOMINGUEZ', 'UMAN', 'TUXTEPEC', 'ZAMORA', 'CORDOBA', 'MONTERREY', 'PENJAMO', 'NOGALES', 'RIOBRAVO', 'CABORCA', 'FRONTERACOAHUILA', 'LOSMOCHIS', 'KANASIN', 'ARRIAGACHIAPAS', 'VALLEHERMOSA', 'SANJOSEITURBIDE', 'MAZATLAN', 'TEHUACAN', 'CHILTEPEC', 'CHILPANCINGODELOSBRAVO'],
            'Latitude': [22.221429, 22.763576, 19.595099, 18.831580, 22.998189, 16.889844, 19.533927, 18.846950, 18.836561, 20.592275, 20.74031, 20.655893, 23.255931, 24.800964, 21.133941, 19.440265, 22.158710, 19.19002, 19.647433, 18.891529, 17.992561, 20.106154, 20.781414, 20.984380, 25.427049, 28.643361, 16.761753, 18.890666,
                            19.271311, 14.679697, 18.833447, 20.054095, 25.845915, 20.76705, 28.431062, 19.736983, 19.500336, 25.717427, 31.239198, 28.165034, 20.13492, 31.785672, 20.488792, 28.721685, 25.594781, 18.88138, 14.950696, 20.842635, 20.646152, 19.799357, 20.313766, 20.958186, 23.786371, 27.541875, 19.863533, 20.531878, 20.380148, 19.891505, 19.641563, 20.566394, 20.576162, 19.971759, 23.215653, 19.132065, 16.801565, 20.707474, 26.128212, 32.6718, 19.943972,
                            21.188758, 18.998997, 18.561445, 31.542897, 20.968175, 16.923231, 21.942294, 32.550529, 16.922181, 26.965938, 17.128621, 20774439, 18.932162, 22.22124, 25.622625, 29.098203, 20.581304, 21.208637, 19.432413, 22.430696, 22.430608, 20.725167, 20.828685, 24.077945, 22.027654, 20.025186, 19.127328, 22.323528, 25.629602, 22.782732, 20.604713, 16.2059, 20.914188, 18.108973, 20.018848, 18.911559, 25.79573, 20.444102, 31.331515, 26.007962, 30.751014, 26.976145, 25.831174, 20.979043, 16.251855, 25.690649, 21.020823, 23.316277, 18.504335, 18.908622, 17.592174],
            'Longitude': [-97.918783, -102.547755, -98.910005, -98.943625, -99.010334, -99.830687, -96.909218, -96.914283, -98.944068, -100.394273, -103.31312, -105.221967, -106.412165, -107.390388, -101.661519, -99.206780, -100.970141, -96.196430, -99.164822, -99.181056, -92.942980, -98.759106, -100.047289, -89.620138, -100.985244, -106.056315, -93.108217, -96.932524,
                            -99.667407, -92.151656, -90.286039, -99.222389, -97.503895, -103.351047, -106.83201, -101.204422, -98.158429, -100.181515, -110.59637, -105.340582, -98.772788, -106.566775, -103.445088, -100.547409, -100.900214, -97.104977, -92.254966, -102.79309, -103.317318, -98.555426, -102.541315, -100.2477, -99.16679, -99.565339, -98.976743, -103.181408, -102.777496, -98.814611, -103.449286, -100.679298, -97.430099, -102.298419, -102.850368, -98.222853, -93.116207, -87.07644, -98.343761, -115.385465, -99.339322,
                            -101.768658, -99.257945, -88.27958, -107.90993, -101.415423, -99.825972, -102.298616, -116.875228, -92.093952, -101.400616, -97.76784, -86.986023, -99.181586, -97.917121, -103.387956, -110.978133, -100.812923, -86.837061, -102.021193, -97.947615, -94.417513, -101.378726, -101.42206, -104.66471, -99.024839, -99.025514, -104.393928, -97.88042, -103.500552, -102.573756, -101.174834, -92.132644, -89.695333, -96.141711, -102.285924, -96.98147, -100.385905, -101.730812, -110.932889, -98.122363, -112.157303, -101.436711, -108.989827, -89.5488, -93.920658, -97.810778, -100.395074, -106.478543, -97414124, -97.047666, -99.51663]
            })
        
        # Agregar coordenadas al DataFrame restante
        data_restante['CiudadDestino'] = data_restante['CiudadDestino'].str.replace(' ', '')
        data_restante = data_restante.merge(Ubicaciones, left_on='CiudadDestino', right_on='City', how='left')

        # Calcular la matriz de distancias haversine
        dist = DistanceMetric.get_metric('haversine')
        coords = np.radians(data_restante[['Latitude', 'Longitude']])
        matriz_distancia = dist.pairwise(coords) * 6371  # Convert to kilometers

        # Listas para almacenar las parejas formadas
        parejas = []

        # Iterar sobre el DataFrame restante
        i = 0
        while i < len(data_restante):
            fila_actual = data_restante.iloc[i]
            for j in range(i + 1, len(data_restante)):
                fila_siguiente = data_restante.iloc[j]
                distancia = matriz_distancia[i, j]
                if distancia <= distancia_maxima and (fila_actual['Horas en patio'] > 23 or fila_siguiente['Horas en patio'] > 23):
                    parejas.append({
                        'Destino1': fila_actual['CiudadDestino'],
                        'Destino2': fila_siguiente['CiudadDestino'],
                        'IdSolicitud1': fila_actual['IdSolicitud'],
                        'IdSolicitud2': fila_siguiente['IdSolicitud'],
                        'Remolque1': fila_actual['Remolque'],
                        'Remolque2': fila_siguiente['Remolque'],
                        'Horas en patio1': fila_actual['Horas en patio'],
                        'Horas en patio2': fila_siguiente['Horas en patio'],
                        'ValorViaje1': fila_actual['ValorViaje'],
                        'ValorViaje2': fila_siguiente['ValorViaje'],
                        'Distancia': distancia
                    })
                    # Eliminar las planas emparejadas del DataFrame original
                    data_restante = data_restante.drop([i, j])
                    data_restante = data_restante.reset_index(drop=True)
                    matriz_distancia = np.delete(matriz_distancia, [i, j], axis=0)
                    matriz_distancia = np.delete(matriz_distancia, [i, j], axis=1)
                    i -= 1
                    break
            i += 1

        # Convertir las parejas a un DataFrame para mostrar el resultado
        parejas_df = pd.DataFrame(parejas)
        
        # Verificar si parejas_df está vacío antes de intentar concatenar las columnas
        if not parejas_df.empty:
            ids_emparejados = pd.concat([parejas_df['IdSolicitud1'], parejas_df['IdSolicitud2']])
            planas_restante = data_restante[~data_restante['IdSolicitud'].isin(ids_emparejados)]
        else:
            planas_restante = data_restante.copy()
        
        return parejas_df, planas_restante
    
    def emparejar_misma_ciudad_menor23(planas):
        # Filtrar las planas con más de 23 horas en el patio
        _, planas_restante = emparejar_destinos_cercanos_mayor23(planas, distancia_maxima=190)
        df_filtrado = planas_restante.copy()
        df_filtrado = planas[planas['Horas en patio'] < 23]
        
        # Listas para almacenar las parejas formadas
        parejas = []
        
        # Agrupar por 'CiudadDestino'
        destinos_agrupados = df_filtrado.groupby('CiudadDestino')
        
        for destino, grupo in destinos_agrupados:
            # Si hay más de una plana en el mismo destino
            if len(grupo) > 1:
                ids = grupo['IdSolicitud'].tolist()
                remolques = grupo['Remolque'].tolist()
                horas = grupo['Horas en patio'].tolist()
                valores = grupo['ValorViaje'].tolist()
                for i in range(0, len(remolques) - 1, 2):
                    if i + 1 < len(remolques):
                        parejas.append({
                            'Destino': destino,
                            'IdSolicitud1': ids[i],
                            'IdSolicitud2': ids[i + 1],
                            'Remolque1': remolques[i],
                            'Remolque2': remolques[i + 1],
                            'Horas en patio1': horas[i],
                            'Horas en patio2': horas[i + 1],
                            'ValorViaje1': valores[i],
                            'ValorViaje2': valores[i + 1]
                        })
 
        # Convertir las parejas a un DataFrame para mostrar el resultado
        parejas_df = pd.DataFrame(parejas)
        
        if not parejas_df.empty:
            ids_emparejados = pd.concat([parejas_df['IdSolicitud1'], parejas_df['IdSolicitud2']])
            planas_restante = planas_restante[~planas_restante['IdSolicitud'].isin(ids_emparejados)]
        else:
            planas_restante = planas_restante.copy()
        
        return parejas_df, planas_restante
    
    def emparejamiento_fin(planas):
        r1,_ = emparejar_misma_ciudad_mayor23(planas)
        r2,_ = emparejar_destinos_cercanos_mayor23(planas, distancia_maxima=200)
        r3,_ = emparejar_misma_ciudad_menor23(planas)
        rf = pd.concat([r1, r2, r3], ignore_index=True)
        
        def combinar_destinos(row):
            if pd.isna(row['Destino1']) and pd.isna(row['Destino2']):
                return row['Destino'] if pd.notna(row['Destino']) else ''
            elif pd.isna(row['Destino1']):
                return row['Destino2']
            elif pd.isna(row['Destino2']):
                return row['Destino1']
            else:
                return f"{row['Destino1']} | {row['Destino2']}"
        # Aplicar la función a cada fila del DataFrame
        rf['Destino'] = rf.apply(combinar_destinos, axis=1)
        rf['Horas en patio'] = rf[['Horas en patio1', 'Horas en patio2']].max(axis=1)
        rf['ValorViaje'] = rf['ValorViaje1'] + rf['ValorViaje2']
        rf= rf.sort_values(by='Horas en patio', ascending=False)
        rf = rf[['Destino', 'Remolque1', 'Remolque2', 'Horas en patio', 'ValorViaje']]
        
        # Reiniciar el índice comenzando desde 1
        rf.reset_index(drop=True, inplace=True)
        rf.index = rf.index + 1
        
        
        # Renombrar las columnas del DataFrame
        rf.rename(columns={
            'Destino': 'Ruta',
            'Remolque1': 'remolque_a',
            'Remolque2': 'remolque_b',
            'Horas en patio': 'Horas en Patio',
            'ValorViaje': 'Monto'
        }, inplace=True)
        
        return rf

    return emparejamiento_fin(planas)