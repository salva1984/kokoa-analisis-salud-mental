# %%
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt

merged_df = pd.read_csv("../data/dataset_completo.csv")

# %%
import pandas as pd

# 1. Calcular cuántas filas quedarían
total_filas = len(merged_df)
filas_sin_nulos = len(merged_df.dropna())
filas_perdidas = total_filas - filas_sin_nulos
porcentaje_perdida = (filas_perdidas / total_filas) * 100

print(f"--- ANÁLISIS DE PÉRDIDA POR DROPNA ---")
print(f"Total de registros originales: {total_filas}")
print(f"Registros que quedarían:       {filas_sin_nulos}")
print(f"Registros a eliminar:          {filas_perdidas}")
print(f"Porcentaje de pérdida:         {porcentaje_perdida:.2f}%")
print("-" * 40)

# 2. Análisis por bloques (Para ver quién es el 'culpable')
# Esto te dice cuántas filas tienen nulos en cada columna específica
nulos_por_columna = merged_df.isnull().sum()
columnas_con_nulos = nulos_por_columna[nulos_por_columna > 0].sort_values(ascending=False)

print("Principales columnas que causan la pérdida:")
print(columnas_con_nulos)

# 3. Tip Pro: ¿Y si solo dropeamos filas que tienen TODO nulo? (Suele ser 0)
filas_vacias_completas = merged_df.isnull().all(axis=1).sum()
if filas_vacias_completas > 0:
    print(f"\nOjo: Hay {filas_vacias_completas} filas que están totalmente vacías.")


# %%
import pandas as pd
import numpy as np

def limpieza_profunda_nhanes(df):
    df_clean = df.copy()

    # 1. BLOQUE ANTROPOMÉTRICO (FÍSICO)
    # Usamos la mediana para no introducir valores extremos en medidas biológicas.
    cols_fisicas = ['BMXWAIST', 'altura_cm', 'peso_libras', 'altura_pulgadas', 'BMXBMI']
    for col in cols_fisicas:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # 2. BLOQUE TABACO (SMQ/SMD)
    # Agregamos SMQ078 a las cualitativas para que se llene con 9 (Sin respuesta)
    sm_qual = ['SMQ020', 'SMQ040', 'SMQ670', 'SMQ078'] # <--- Corregido
    sm_quant = ['SMQ852_dias', 'SMQ848', 'SMD030', 'SMD641', 'SMD650']
    
    for col in sm_qual: 
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(9)
            
    for col in sm_quant: 
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(-1)

    # 3. BLOQUE PESO Y PERCEPCIÓN
    # Tratamos los nulos como una categoría propia de "Sin Información"
    cols_percepcion = ['conductas_riesgo_peso', 'percibe_peso_como', 'intento_bajar_peso']
    for col in cols_percepcion:
        df_clean[col] = df_clean[col].fillna('Sin_Dato')

    # 4. BLOQUE OCUPACIÓN (OCQ/OCD)
    # Códigos de industria/trabajo (cat) y Horas/Duración (num)
    oc_cat = ['OCD391', 'OCD392', 'OCQ210', 'OCD390G', 'OCQ260', 'OCD231', 'OCD241', 'OCD150']
    oc_num = ['OCQ180', 'OCD395']
    for col in oc_cat: df_clean[col] = df_clean[col].fillna(99)
    for col in oc_num: df_clean[col] = df_clean[col].fillna(-1)

    # 5. BLOQUE ACTIVIDAD FÍSICA Y SEDENTARISMO
    act_qual = ['transporte_activo_bici_caminar', 'ejercicio_moderado_recreativo', 
                'ejercicio_vigoroso_recreativo', 'trabajo_moderado', 'trabajo_vigoroso']
    act_num = ['horas_sedentario_dia', 'minutos_sedentario_dia']
    for col in act_qual: df_clean[col] = df_clean[col].fillna('No_Informa')
    for col in act_num: df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # 6. BLOQUE DROGAS (DUQ)
    duq_qual = ['DUQ200', 'DUQ370', 'DUQ430', 'DUQ240']
    duq_quant = ['DUQ410', 'DUQ320', 'DUQ210', 'DUQ360', 'DUQ230', 'DUQ280']
    for col in duq_qual: df_clean[col] = df_clean[col].fillna(9)
    for col in duq_quant: df_clean[col] = df_clean[col].fillna(-1)

    # 7. BLOQUE ALCOHOL
    alc_qual = ['Consumio_Alcohol']
    alc_num = ['Promedio_Tragos_Dia', 'Dias_Consumo_Anual', 
               'Dias_Con_+4_5_Tragos_En_2hrs', 'Dias_Alto_Consumo_Anual']
    for col in alc_qual: df_clean[col] = df_clean[col].fillna('Missing')
    for col in alc_num: df_clean[col] = df_clean[col].fillna(-1)

    return df_clean

# Aplicar a tu DataFrame
df_final = limpieza_profunda_nhanes(merged_df)

# %%

# Verificación final
nulos_restantes = df_final.isnull().sum().sum()
print(f"Total de nulos después del arreglo: {nulos_restantes}")
print(f"Registros salvados: {len(df_final)}")
# %%

# 1. Calcular cuántas filas quedarían
total_filas = len(df_final)
filas_sin_nulos = len(df_final.dropna())
filas_perdidas = total_filas - filas_sin_nulos
porcentaje_perdida = (filas_perdidas / total_filas) * 100

print(f"--- ANÁLISIS DE PÉRDIDA POR DROPNA ---")
print(f"Total de registros originales: {total_filas}")
print(f"Registros que quedarían:       {filas_sin_nulos}")
print(f"Registros a eliminar:          {filas_perdidas}")
print(f"Porcentaje de pérdida:         {porcentaje_perdida:.2f}%")
print("-" * 40)

# 2. Análisis por bloques (Para ver quién es el 'culpable')
# Esto te dice cuántas filas tienen nulos en cada columna específica
nulos_por_columna = df_final.isnull().sum()
columnas_con_nulos = nulos_por_columna[nulos_por_columna > 0].sort_values(ascending=False)

print("Principales columnas que causan la pérdida:")
print(columnas_con_nulos)
# %%
df_final.drop(columns=['Unnamed: 0','RIDAGEYR'], inplace=True)
df_final.info()
df_final.to_csv("../data/processed/dataset_limpio.csv", index=False)
# %%