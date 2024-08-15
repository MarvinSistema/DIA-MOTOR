import redis
import pandas as pd
import pickle
import os
import gdown
import io


redis_url = os.getenv('REDIS_URL')#'redis://localhost:6379'
redis_client = redis.StrictRedis.from_url(redis_url)

def actualizar_cache():
    cache_key = 'seguimiento_data'

    # Verificar si los datos ya están en caché antes de intentar actualizarlos
    if not redis_client.get(cache_key):
        print("El cache no está precalentado. Leyendo y actualizando...")
        url = 'https://drive.google.com/uc?id=1h3oynOXp11tKAkNmq4SkjBR8q_ZyJa2b'
        path = gdown.download(url, output=None, quiet=False)  # Guarda el archivo temporalmente
        # Abrir el archivo temporal en modo binario
        with open(path, 'rb') as f:
            data = io.BytesIO(f.read())  # Leer los datos del archivo y pasarlos a BytesIO
        # Leer el archivo desde el buffer directamente en un DataFrame
        df = pd.read_excel(data)
        df = df.sort_values(by='fecha de salida', ascending=False, na_position='last')
        df = df.groupby('Remolque')['fecha de salida'].max().reset_index()

        # Guardar en caché
        redis_client.setex(cache_key, 900, pickle.dumps(df))

    else:
        print("El cache ya está precalentado. No es necesario leer el archivo Excel nuevamente.")


actualizar_cache()