# %%
import pandas as pd
import numpy as np

# ==========================================
# 1. CARGA DE DATOS Y MERGE INICIAL
# ==========================================
duq = pd.read_csv("../data/interim/duq_h_columnas.csv")
df_dep = pd.read_csv("../data/processed/dpq_limpio.csv")
df_demo = pd.read_csv("../data/interim/demo_h_columnas.csv")

# Filtrar depresión nula
df_dep_clean = df_dep.dropna(subset=['score_depresion'])

# Merge de los 3 datasets de una vez
df = df_dep_clean[['SEQN', 'score_depresion']].merge(
    df_demo[['SEQN', 'RIDAGEYR']], on='SEQN', how='inner'
).merge(
    duq, on='SEQN', how='left'
)

# ==========================================
# 2. LIMPIEZA DE INVÁLIDOS (Drop)
# ==========================================
# Agrupamos todos los códigos de error que obligan a eliminar la fila en una sola máscara
mascara_eliminar = (
    df['DUQ210'].isin([777, 999]) |
    df['DUQ230'].isin([777, 999]) |
    df['DUQ240'].isin([7, 9]) |
    df['DUQ320'].isin([77, 99]) |
    df['DUQ360'].isin([77, 99]) |
    df['DUQ370'].isin([7, 9])
)
df = df[~mascara_eliminar].copy()

# ==========================================
# 3. REGLAS POR EDAD (No aplica)
# ==========================================
mayor_59 = df['RIDAGEYR'] > 59
mayor_69 = df['RIDAGEYR'] > 69

# > 59 años
df.loc[mayor_59, 'DUQ200'] = 3
df.loc[df['DUQ200'] == 3, ['DUQ210', 'DUQ230']] = -2

cols_59_na = ['DUQ280', 'DUQ320', 'DUQ360', 'DUQ430']
for col in cols_59_na:
    df.loc[mayor_59 & df[col].isna(), col] = -2

# > 69 años
df.loc[mayor_69, 'DUQ240'] = 3
cols_69_na = ['DUQ370', 'DUQ410']
for col in cols_69_na:
    df.loc[mayor_69 & df[col].isna(), col] = -2

# ==========================================
# 4. FLUJO DE PREGUNTAS (Lógica de "Nunca" / "Sí, pero no reciente")
# ==========================================
# -- Marihuana --
# Lógica original: A los [2,7,9] se les ponía 0, y luego a los [2] se les sobreescribía con -1 en DUQ210
df.loc[df['DUQ200'].isin([2, 7, 9]), ['DUQ210', 'DUQ230']] = 0
df.loc[df['DUQ200'] == 2, 'DUQ210'] = -1

# Sí consumió, pero nulo en último mes -> 0
df.loc[(df['DUQ200'] == 1) & df['DUQ230'].isna(), 'DUQ230'] = 0

# -- Drogas Pesadas --
# Nunca usó -> 0 días de consumo
df.loc[df['DUQ240'] == 2, ['DUQ280', 'DUQ320', 'DUQ360']] = 0
# Sí usó, pero nulo en último mes -> 0 días de consumo
uso_pesadas_na = (df['DUQ240'] == 1)
for col in ['DUQ280', 'DUQ320', 'DUQ360']:
    df.loc[uso_pesadas_na & df[col].isna(), col] = 0

# -- Inyectables --
df.loc[df['DUQ370'] == 2, 'DUQ410'] = 0

# -- Rehabilitación --
df.loc[(df['DUQ200'] == 2) & (df['DUQ240'] == 2), 'DUQ430'] = 2

# ==========================================
# 5. MANEJO DE GRUPO 4 Y NULOS RESTANTES
# ==========================================
# Convertimos los "No sabe / Rehusó" específicos a -4 para no perder la fila
df.loc[df['DUQ410'] == 99, 'DUQ410'] = -4
df.loc[df['DUQ430'] == 7, 'DUQ430'] = -4

# Fillna para categorías base
df['DUQ200'] = df['DUQ200'].fillna(4)
df['DUQ240'] = df['DUQ240'].fillna(4)

# Todo lo que quede nulo en las columnas DUQ se vuelve -4.0 (Esto cubre tu "Grupo 4" 
# y cualquier otro nulo huérfano asegurando que no se pierdan las filas con depresión)
cols_duq = [c for c in df.columns if c.startswith('DUQ')]
df[cols_duq] = df[cols_duq].fillna(-4.0)

# ==========================================
# 6. VERIFICACIÓN Y EXPORTACIÓN
# ==========================================
print("--- Estado Final del Dataset ---")
df.info()

print("\nDistribución de Rehabilitación (DUQ430):")
print(df['DUQ430'].value_counts().sort_index())

df.to_csv("../data/processed/duq_limpio.csv", index=False)