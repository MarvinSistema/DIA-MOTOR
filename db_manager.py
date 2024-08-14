import pandas as pd
from sqlalchemy import create_engine
from config import DB_CONFIG, DB_CONFIG_PRO, DB_CONFIG_DIA




def create_engine_db():
    """Crea y devuelve un motor de base de datos usando SQLAlchemy."""
    connection_str = f"mssql+pyodbc://{DB_CONFIG['username']}:{DB_CONFIG['password']}@{DB_CONFIG['server']}/{DB_CONFIG['database']}?driver={DB_CONFIG['driver']}"
    try:
        engine = create_engine(connection_str)
        print("Conexión exitosa")
        return engine
    except Exception as e:
        print(f"Error al conectar: {e}")
        return None

def create_engine_db_PRO():
    """Crea y devuelve un motor de base de datos usando SQLAlchemy."""
    connection_str = f"mssql+pyodbc://{DB_CONFIG_PRO['username']}:{DB_CONFIG_PRO['password']}@{DB_CONFIG_PRO['server']}/{DB_CONFIG_PRO['database']}?driver={DB_CONFIG_PRO['driver']}"
    try:
        engine = create_engine(connection_str)
        print("Conexión exitosa")
        return engine
    except Exception as e:
        print(f"Error al conectar: {e}")
        return None

def create_engine_db_DIA():
    """Crea y devuelve un motor de base de datos usando SQLAlchemy."""
    connection_str = f"mssql+pyodbc://{DB_CONFIG_DIA['username']}:{DB_CONFIG_DIA['password']}@{DB_CONFIG_DIA['server']}/{DB_CONFIG_DIA['database']}?driver={DB_CONFIG_DIA['driver']}"
    try:
        engine = create_engine(connection_str)
        print("Conexión exitosa")
        return engine
    except Exception as e:
        print(f"Error al conectar: {e}")
        return None


def fetch_data(query):
    """Ejecuta una consulta y devuelve los resultados como un DataFrame de Pandas."""
    engine = create_engine_db()
    if engine is not None:
        try:
            result = pd.read_sql(query, engine)
        except Exception as e:
            print(f"Error al realizar la consulta: {e}")
            result = pd.DataFrame()
        return result
    return pd.DataFrame()  # Devuelve un DataFrame vacío si la conexión falla

def fetch_data_PRO(query):
    """Ejecuta una consulta y devuelve los resultados como un DataFrame de Pandas."""
    engine = create_engine_db_PRO()
    if engine is not None:
        try:
            result = pd.read_sql(query, engine)
        except Exception as e:
            print(f"Error al realizar la consulta: {e}")
            result = pd.DataFrame()
        return result
    return pd.DataFrame()  # Devuelve un DataFrame vacío si la conexión falla

def fetch_data_DIA(query):
    """Ejecuta una consulta y devuelve los resultados como un DataFrame de Pandas."""
    engine = create_engine_db_DIA()
    if engine is not None:
        try:
            result = pd.read_sql(query, engine)
        except Exception as e:
            print(f"Error al realizar la consulta: {e}")
            result = pd.DataFrame()
        return result
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
    return pd.DataFrame()  