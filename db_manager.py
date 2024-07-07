import pandas as pd
import pyodbc
from config import DB_CONFIG, DB_CONFIG_PRO, DB_CONFIG_DIA

def create_engine_db():
    """Crea y devuelve una conexión directa con la base de datos usando pyodbc."""
    connection_str = f"DRIVER={{{DB_CONFIG['driver']}}};SERVER={DB_CONFIG['server']};DATABASE={DB_CONFIG['database']};UID={DB_CONFIG['username']};PWD={DB_CONFIG['password']}"
    try:
        connection = pyodbc.connect(connection_str, autocommit=True)
        print("Conexión exitosa")
        return connection
    except Exception as e:
        print(f"Error al conectar: {e}")
        return None

def create_engine_db_PRO():
    """Crea y devuelve una conexión directa con la base de datos usando pyodbc."""
    connection_str = f"DRIVER={{{DB_CONFIG_PRO['driver']}}};SERVER={DB_CONFIG_PRO['server']};DATABASE={DB_CONFIG_PRO['database']};UID={DB_CONFIG_PRO['username']};PWD={DB_CONFIG_PRO['password']}"
    try:
        connection = pyodbc.connect(connection_str, autocommit=True)
        print("Conexión exitosa")
        return connection
    except Exception as e:
        print(f"Error al conectar: {e}")
        return None
    
def create_engine_db_DIA():
    """Crea y devuelve una conexión directa con la base de datos usando pyodbc."""
    connection_str = f"DRIVER={{{DB_CONFIG['driver']}}};SERVER={DB_CONFIG_DIA['server']};DATABASE={DB_CONFIG_DIA['database']};UID={DB_CONFIG['username']};PWD={DB_CONFIG['password']}"
    try:
        connection = pyodbc.connect(connection_str, autocommit=True)
        print("Conexión exitosa")
        return connection
    except Exception as e:
        print(f"Error al conectar: {e}")
        return None

def fetch_data(query):
    """Ejecuta una consulta y devuelve los resultados como un DataFrame de Pandas."""
    connection = create_engine_db()
    if connection is not None:
        try:
            result = pd.read_sql(query, connection)
        except Exception as e:
            print(f"Error al realizar la consulta: {e}")
            result = pd.DataFrame()
        finally:
            connection.close()  # Asegurarse de cerrar la conexión
        return result
    return pd.DataFrame()  # Devuelve un DataFrame vacío si la conexión falla

def fetch_data_PRO(query):
    """Ejecuta una consulta y devuelve los resultados como un DataFrame de Pandas."""
    connection = create_engine_db_PRO()
    if connection is not None:
        try:
            result = pd.read_sql(query, connection)
        except Exception as e:
            print(f"Error al realizar la consulta: {e}")
            result = pd.DataFrame()
        finally:
            connection.close()  # Asegurarse de cerrar la conexión
        return result
    return pd.DataFrame()  # Devuelve un DataFrame vacío si la conexión falla
   
def fetch_data_DIA(query):
    """Ejecuta una consulta y devuelve los resultados como un DataFrame de Pandas."""
    connection = create_engine_db_DIA()
    if connection is not None:
        try:
            result = pd.read_sql(query, connection)
        except Exception as e:
            print(f"Error al realizar la consulta: {e}")
            result = pd.DataFrame()
        finally:
            connection.close()  # Asegurarse de cerrar la conexión
        return result
    return pd.DataFrame()  #