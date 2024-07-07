import os
from dotenv import load_dotenv

# Carga las variables de entorno desde el archivo .env
load_dotenv()

# Asegúrate de que las variables de entorno estén definidas o utiliza valores predeterminados
DB_CONFIG = {
    "server": os.getenv('DB_SERVER', 'default_server'),
    "database": os.getenv('DB_DATABASE', 'default_database'),
    "username": os.getenv('DB_USERNAME', 'default_user'),
    "password": os.getenv('DB_PASSWORD', 'default_password'),
    "driver": os.getenv('DB_DRIVER', 'ODBC Driver 17 for SQL Server').replace('+', ' ')
}

DB_CONFIG_PRO = {
    "server": os.getenv('DB_PRO_SERVER', 'default_pro_server'),
    "database": os.getenv('DB_PRO_DATABASE', 'default_pro_database'),
    "username": os.getenv('DB_PRO_USERNAME', 'default_pro_user'),
    "password": os.getenv('DB_PRO_PASSWORD', 'default_pro_password'),
    "driver": os.getenv('DB_PRO_DRIVER', 'ODBC Driver 17 for SQL Server').replace('+', ' ')
}

DB_CONFIG_DIA = {
    "server": os.getenv('DB_SERVER_DIA', 'default_server'),
    "database": os.getenv('DB_DATABASE_DIA', 'default_database'),
    "username": os.getenv('DB_USERNAME', 'default_user'),
    "password": os.getenv('DB_PASSWORD', 'default_password'),
    "driver": os.getenv('DB_DRIVER', 'ODBC Driver 17 for SQL Server').replace('+', ' ')
}
