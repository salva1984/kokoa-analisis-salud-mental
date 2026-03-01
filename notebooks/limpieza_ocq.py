# %%
# Cargar datos OCQ
import pandas as pd
import numpy as np
ocq = pd.read_csv("../data/interim/ocq_h_columnas.csv")
df_demo = pd.read_csv("../data/interim/demo_h_columnas.csv")
df_edad = df_demo[['SEQN', 'RIDAGEYR']].copy()
ocq = ocq.merge(df_edad,on='SEQN',how='inner')
ocq = ocq[ocq['RIDAGEYR']>=18]

OCQ_COLS = [
    "SEQN",
    'OCD150',
    "OCQ180",   # Horas trabajadas la semana pasada
    "OCQ210",   # Trabaja ≥35h (1=Sí, 2=No)
    "OCD231",   # Industria actual
    "OCD241",   # Ocupación actual
    "OCQ260",   # Tipo de empleo
    "OCD270",   # meses en el trabajo actual
    "OCD390G",  # Tipo de trabajo más largo
    "OCD391",   # Industria trabajo más largo
    "OCD392",   # Ocupación trabajo más largo
    "OCD395"    # Meses en el trabajo más largo
]

ocq = ocq[OCQ_COLS].copy()
ocq.head()

not_at_work = ocq['OCD150'] == 2
# saltas esta pregunta
ocq.loc[not_at_work, 'OCQ180'] = 0

looking_for_work_or_not_working = ocq['OCD150'].isin([3, 4])
# saltar preguntas
ocq.loc[looking_for_work_or_not_working, 'OCQ180'] = 0
ocq.loc[looking_for_work_or_not_working, 'OCQ210'] = 2 # no
ocq.loc[looking_for_work_or_not_working, 'OCD231'] = 0
ocq.loc[looking_for_work_or_not_working, 'OCD241'] = 0
ocq.loc[looking_for_work_or_not_working, 'OCQ260'] = 0



# 1. Definimos la condición (máscara)
same_as_current = ocq['OCD390G'] == 2
# Mapeo: OCD231 -> OCD391 (Industria)
ocq.loc[same_as_current, 'OCD391'] = ocq.loc[same_as_current, 'OCD231']
# Mapeo: OCD241 -> OCD392 (Ocupación)
ocq.loc[same_as_current, 'OCD392'] = ocq.loc[same_as_current, 'OCD241']
# Mapeo: OCD270 -> OCD395 (Meses en el trabajo)
ocq.loc[same_as_current, 'OCD395'] = ocq.loc[same_as_current, 'OCD270']


# Definimos las condiciones 
never_worked = ocq['OCD390G'] == 4
invalid = ocq['OCD390G'].isin([9, 7])
# Usamos '|' para el "OR" lógico entre Series
skip = never_worked | invalid
# Aplicamos los cambios con .loc
ocq.loc[skip, 'OCD391'] = 0
ocq.loc[skip, 'OCD392'] = 0
ocq.loc[skip, 'OCD395'] = 0

# si trabajas mas de 35 horas, no aplicas (respondes si)
filtro_mas_35h = ocq['OCQ180'] >= 35
ocq.loc[filtro_mas_35h, 'OCQ210'] = 1 #si

# ocd270 no estaba evaluada, dropear columna
ocq.drop(columns=['OCD270'], inplace=True)

# %%

# limpieza valores invalidos
invalid_codes = [77777, 99999]
ocq = ocq[~ocq['OCQ180'].isin(invalid_codes)].copy()

invalid_ocq210 = [7, 9]
ocq = ocq[~ocq['OCQ210'].isin(invalid_ocq210)].copy()

invalid_ocq260 = [77, 99]
ocq = ocq[~ocq['OCQ260'].isin(invalid_ocq260)].copy()

invalid_ocd390g = [7, 9]
ocq = ocq[~ocq['OCD390G'].isin(invalid_ocd390g)].copy()

invalid_ocd395 = [77777, 99999]
ocq = ocq[~ocq['OCD395'].isin(invalid_ocd395)].copy()

# revisamos nulos
ocq.isna().sum()

# %%
# Filtramos las filas que tienen al menos un valor NaN en cualquier columna
filas_con_nulos = ocq[ocq.isna().any(axis=1)]

# Mostramos el resultado
filas_con_nulos 
# %%
# son pocos nulos, dropeamos
ocq.dropna(inplace=True)
ocq.to_csv("../data/processed/ocq_limpio.csv", index=False)