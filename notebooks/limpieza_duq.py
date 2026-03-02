# %%
import pandas as pd
import numpy as np

duq = pd.read_csv("../data/interim/duq_h_columnas.csv")
duq.info()
# %%

# flujo de preguntas
not_yes = duq['DUQ200'].isin([2,7,9])
duq.loc[not_yes, 'DUQ210'] = 0
duq.loc[not_yes, 'DUQ230'] = 0

# mayores de 59 no aplican
df_dep = pd.read_csv("../data/processed/dpq_limpio.csv")
df_dep_clean = df_dep.dropna(subset=['score_depresion'])
df_demo = pd.read_csv("../data/interim/demo_h_columnas.csv")
df = pd.merge(
    df_dep_clean[['SEQN', 'score_depresion']], 
    df_demo[['SEQN', 'RIDAGEYR']], 
    on='SEQN', 
    how='inner'
)

df = pd.merge(df, duq, on='SEQN', how='left')

df.loc[df['RIDAGEYR'] > 59, 'DUQ200'] = 3 # no aplica

# 1. Primero limpiamos los inválidos reales (777 y 999)
df = df[~df['DUQ210'].isin([777, 999])].copy()

# 2. Manejamos a los que NUNCA fumaron (DUQ200 == 2)
# Les ponemos -1 para que el modelo sepa que "No aplica"
df.loc[df['DUQ200'] == 2, 'DUQ210'] = -1

# 3. Manejamos a los Adultos Mayores (> 59)
# Les ponemos -2 (ya que les pusimos 3 en DUQ200)
df.loc[df['DUQ200'] == 3, 'DUQ210'] = -2

# 1. Limpieza de inválidos (777, 999) - Aunque el conteo dice 0, por seguridad:
df = df[~df['DUQ230'].isin([777, 999])].copy()

# 2. Los que NUNCA han fumado (DUQ200 == 2) -> 0 días en el último mes
df.loc[df['DUQ200'] == 2, 'DUQ230'] = 0

# 3. Los que SÍ han fumado alguna vez (DUQ200 == 1) pero el dato es nulo
# Esto significa que no han fumado en el último mes según el flujo de la encuesta.
df.loc[(df['DUQ200'] == 1) & (df['DUQ230'].isna()), 'DUQ230'] = 0

# 4. Los Adultos Mayores (> 59 años)
# Usamos -2 para mantenerlos como "Categoría aparte" y no bajar el promedio de consumo real
df.loc[df['DUQ200'] == 3, 'DUQ230'] = -2

# 1. Limpieza de inválidos (7: Refused, 9: Don't know)
# Los eliminamos porque no podemos saber si consumieron o no.
df = df[~df['DUQ240'].isin([7, 9])].copy()

# 2. Aplicamos la lógica de "No aplica por edad" (> 69 años)
# Nota que ahora el límite es 69, no 59.
df.loc[(df['RIDAGEYR'] > 69), 'DUQ240'] = 3

# 3.
# A los que quedaron con NaN en DUQ200 (y están en rango de edad), 
# les asignamos la categoría 4: "No quiso/pudo responder"
df['DUQ200'] = df['DUQ200'].fillna(4)
# Y lo mismo para la cocaína/heroína
df['DUQ240'] = df['DUQ240'].fillna(4)

# 1. Los que NUNCA han usado drogas pesadas (DUQ240 == 2)
# Su respuesta lógica a "cuántos días al mes" es 0.
df.loc[df['DUQ240'] == 2, 'DUQ280'] = 0

# 2. Los que SI han usado alguna vez (DUQ240 == 1) pero tienen nulo en DUQ280
# El flujo de la encuesta indica que si es nulo aquí, es porque no usaron en los últimos 30 días.
df.loc[(df['DUQ240'] == 1) & (df['DUQ280'].isna()), 'DUQ280'] = 0

# 3. Adultos Mayores (> 59 años)
# OJO: Aunque DUQ240 era hasta los 69, DUQ280 se corta a los 59 según el Target.
df.loc[(df['RIDAGEYR'] > 59) & (df['DUQ280'].isna()), 'DUQ280'] = -2

# 4. El "Grupo 4" (Los que no respondieron nada de drogas pero tienen depresión)
df.loc[(df['DUQ200'] == 4) & (df['DUQ280'].isna()), 'DUQ280'] = -4


# 1. Identificamos todas las columnas que empiezan con DUQ
cols_duq = [c for c in df.columns if c.startswith('DUQ')]

# 2. Para el Grupo 4 (los que tienen 4.0 en DUQ200), 
# ponemos -4.0 en todas las columnas de drogas que sigan siendo nulas
df.loc[df['DUQ200'] == 4, cols_duq] = df.loc[df['DUQ200'] == 4, cols_duq].fillna(-4.0)


# 320
# 1. Por seguridad, limpiamos códigos de error (aunque el conteo diga 0)
df = df[~df['DUQ320'].isin([77, 99])].copy()

# 2. Los que NUNCA han usado drogas pesadas (DUQ240 == 2) -> 0 días
df.loc[df['DUQ240'] == 2, 'DUQ320'] = 0

# 3. Los que SÍ han usado drogas pesadas alguna vez (DUQ240 == 1) 
# pero el dato es nulo en DUQ320 (significa que no usaron heroína este mes)
df.loc[(df['DUQ240'] == 1) & (df['DUQ320'].isna()), 'DUQ320'] = 0

# 4. Adultos Mayores (> 59 años) -> Código -2
df.loc[(df['RIDAGEYR'] > 59) & (df['DUQ320'].isna()), 'DUQ320'] = -2

# 5. Grupo 4 (Los que no respondieron nada de drogas) -> Código -4
df.loc[(df['DUQ200'] == 4) & (df['DUQ320'].isna()), 'DUQ320'] = -4

# %%
# 360
# 1. Por seguridad, eliminamos códigos de error (77, 99)
df = df[~df['DUQ360'].isin([77, 99])].copy()

# 2. Los que NUNCA han usado drogas pesadas (DUQ240 == 2) -> 0 días
# Si nunca han probado coca/heroin/meta, no la usaron este mes.
df.loc[df['DUQ240'] == 2, 'DUQ360'] = 0

# 3. Los que SÍ han usado drogas pesadas alguna vez (DUQ240 == 1)
# pero tienen nulo en DUQ360 (significa que no usaron meta en los últimos 30 días)
df.loc[(df['DUQ240'] == 1) & (df['DUQ360'].isna()), 'DUQ360'] = 0

# 4. Adultos Mayores (> 59 años) -> Código -2
# Nota: DUQ240 era hasta los 69, pero esta frecuencia se corta a los 59.
df.loc[(df['RIDAGEYR'] > 59) & (df['DUQ360'].isna()), 'DUQ360'] = -2

# 5. Grupo 4 (Los que no respondieron nada de drogas pero tienen depresión) -> Código -4
df.loc[(df['DUQ200'] == 4) & (df['DUQ360'].isna()), 'DUQ360'] = -4

# 370
# 1. Limpieza de códigos de error (7: Refused, 9: Don't know)
# Como son solo 4 casos en total, los eliminamos para no meter ruido
df = df[~df['DUQ370'].isin([7, 9])].copy()

# 2. Aplicamos la lógica de "No aplica por edad" (> 69 años)
# Nota: Aquí el límite es 69, no 59.
df.loc[(df['RIDAGEYR'] > 69) & (df['DUQ370'].isna()), 'DUQ370'] = -2

# 3. Grupo 4 (Los que no respondieron nada de drogas pero tienen depresión)
# Usamos el código -4 para identificar que el dato falta por "silencio" del participante
df.loc[(df['DUQ200'] == 4) & (df['DUQ370'].isna()), 'DUQ370'] = -4

# 4. Verificación de nulos restantes
# Si queda alguien entre 18-69 con NaN, lo marcamos como -4 también 
# para mantener las 5355 filas con depresión
df['DUQ370'] = df['DUQ370'].fillna(-4)

# %%
# 1. Manejo de inválidos (99: Don't know)
# Solo hay 1 caso; lo marcamos como -4 (Sin datos) para no perder la fila
df.loc[df['DUQ410'] == 99, 'DUQ410'] = -4

# 2. Los que NUNCA se han inyectado (DUQ370 == 2)
# Su valor lógico de "cuántas veces en la vida" es 0
df.loc[df['DUQ370'] == 2, 'DUQ410'] = 0

# 3. Adultos Mayores (> 69 años)
# Siguen la regla del -2 (No aplica por diseño)
df.loc[(df['RIDAGEYR'] > 69) & (df['DUQ410'].isna()), 'DUQ410'] = -2

# 4. Grupo 4 y nulos por error (Los que no respondieron drogas pero tienen depresión)
# Aplicamos el -4 para mantener la integridad de los 5,355 registros
df.loc[df['DUQ410'].isna(), 'DUQ410'] = -4

# 5. Verificación de la escala ordinal resultante
print("Escala de severidad de inyección (DUQ410):")
print(df['DUQ410'].value_counts().sort_index())
#%%
# 1. Manejo de inválidos (7: Refused)
# Solo hay 1 caso; lo marcamos como -4 (Sin datos)
df.loc[df['DUQ430'] == 7, 'DUQ430'] = -4

# 2. Los que NUNCA han usado ninguna droga
# Si DUQ200 == 2 (No marihuana) Y DUQ240 == 2 (No pesadas), 
# su respuesta lógica a rehabilitación es 2 (No).
df.loc[(df['DUQ200'] == 2) & (df['DUQ240'] == 2), 'DUQ430'] = 2

# 3. Adultos Mayores (> 59 años)
# Volvemos al límite de 59. A los mayores de esa edad les ponemos -2.
df.loc[(df['RIDAGEYR'] > 59) & (df['DUQ430'].isna()), 'DUQ430'] = -2

# 4. Grupo 4 y nulos por error (Sin datos de drogas pero con depresión)
# Aplicamos el -4 para mantener nuestras 5,355 filas.
df.loc[df['DUQ430'].isna(), 'DUQ430'] = -4

# 5. Verificación final de categorías
print("Distribución de Rehabilitación (DUQ430):")
print(df['DUQ430'].value_counts().sort_index())

# %%
df.info()

# %%
# 1. Identificamos las columnas con los 2 nulos
cols_discrepantes = ['DUQ280', 'DUQ320', 'DUQ360']

# 2. Para estas columnas, si el valor es nulo, aplicamos la lógica de edad
# Si son > 59 años, les corresponde el -2
df.loc[(df['RIDAGEYR'] > 59) & (df[cols_discrepantes[0]].isna()), cols_discrepantes] = -2

# 3. Por si acaso esos 2 nulos fueran del "Grupo 4" (sin datos generales)
# les asignamos el -4
for col in cols_discrepantes:
    df[col] = df[col].fillna(-4.0)

# 4. Verificación final
print("--- Estado Final del Dataset ---")
print(df.info())

# %%
df.to_csv("../data/processed/duq_limpio.csv", index=False)
#
# %%
