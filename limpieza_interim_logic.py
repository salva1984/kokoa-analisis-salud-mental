import pandas as pd
import numpy as np

# Cargar el dataset
df = pd.read_csv('data/interim_con_depresion.csv')

# 1. Limpieza de valores cercanos a cero (ej. 5.39e-79) -> 0
# Seleccionamos las columnas numéricas para esta operación
cols_numericas = df.select_dtypes(include=[np.number]).columns
df[cols_numericas] = df[cols_numericas].apply(lambda x: x.where(x > 1e-10, 0))

# 2. Convertir códigos de "No sabe" (999) o "Rechazado" (777) a NaN
# También consideramos 7777 y 9999 que son comunes en SMQ
df = df.replace([777, 999, 7777, 9999], np.nan)

# 3. Estandarización de Alcohol (ALQ)
# Factores: {1: 52 (semana), 2: 12 (mes), 3: 1 (año)}
factores_alq = {1: 52, 2: 12, 3: 1}

# Estandarizar ALQ120 (Frecuencia de consumo)
df['ALQ120_dias_anio'] = df['ALQ120Q'] * df['ALQ120U'].map(factores_alq)
# Si la cantidad (Q) es 0, el resultado es 0 aunque la unidad (U) sea NaN
df.loc[df['ALQ120Q'] == 0, 'ALQ120_dias_anio'] = 0

# Estandarizar ALQ141 (Días de alto consumo 4-5 tragos)
df['ALQ141_dias_anio'] = df['ALQ141Q'] * df['ALQ141U'].map(factores_alq)
df.loc[df['ALQ141Q'] == 0, 'ALQ141_dias_anio'] = 0

# 4. Estandarización de Tabaco (SMQ)
# Factores: {1: 1 (día), 2: 7 (semana), 3: 30.4 (mes)}
factores_smq = {1: 1, 2: 7, 3: 30.4}

df['SMQ852_dias_total'] = df['SMQ852Q'] * df['SMQ852U'].map(factores_smq)
df.loc[df['SMQ852Q'] == 0, 'SMQ852_dias_total'] = 0

# Guardar el dataset procesado
output_path = 'data/interim_con_depresion.csv' # El usuario pidió procesar este archivo, asumo que quiere sobreescribirlo o mantenerlo actualizado
df.to_csv(output_path, index=False)

print(f"Transformación completada con éxito. Archivo actualizado: {output_path}")
print(f"ALQ120_dias_anio (mean): {df['ALQ120_dias_anio'].mean():.2f}")
print(f"ALQ141_dias_anio (mean): {df['ALQ141_dias_anio'].mean():.2f}")
print(f"SMQ852_dias_total (mean): {df['SMQ852_dias_total'].mean():.2f}")
