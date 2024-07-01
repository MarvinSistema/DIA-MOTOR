FROM python:3.9-slim

# Instalar dependencias necesarias
RUN apt-get update && apt-get install -y curl gnupg2

# Configurar repositorio de Microsoft para SQL Server
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list

# Instalar el driver ODBC de SQL Server
RUN apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql17

# Limpiar cache de APT
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto
EXPOSE $PORT

# Comando para ejecutar la aplicación
CMD gunicorn app:app --bind 0.0.0.0:$PORT

