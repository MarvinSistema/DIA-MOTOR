import redis
import pandas as pd
import pickle
from datetime import timedelta
import os

redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
redis_client = redis.StrictRedis.from_url(redis_url)

def actualizar_cache():
    cache_key = 'seguimiento_data'
    cache_expiration = timedelta(minutes=15)

    # Verificar si los datos ya están en caché antes de intentar actualizarlos
    if not redis_client.get(cache_key):
        print("El cache no está precalentado. Leyendo y actualizando...")
        df = pd.read_excel('seguimiento_ternium.xlsx')
        df = df.sort_values(by='fecha de salida', ascending=False, na_position='last')
        df = df.groupby('Remolque')['fecha de salida'].max().reset_index()

        # Guardar en caché
        redis_client.setex(cache_key, cache_expiration, pickle.dumps(df))
    else:
        print("El cache ya está precalentado. No es necesario leer el archivo Excel nuevamente.")

actualizar_cache()