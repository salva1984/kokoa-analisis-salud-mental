# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("../data/dataset_completo.csv")
# 1. Seleccionar solo columnas numéricas y booleanas
cols_analisis = df.select_dtypes(include=[np.number, bool]).columns
df_numerico = df[cols_analisis]

# 1. Calcular la matriz de correlación absoluta
matriz_corr_abs = df_numerico.corr(method='spearman').abs()
# 1. Crear una lista para almacenar las parejas con alta correlación
parejas_redundantes = []

# 2. Iterar sobre la matriz que calculaste antes
for i in range(len(matriz_corr_abs.columns)):
    for j in range(i + 1, len(matriz_corr_abs.columns)):
        coeficiente = matriz_corr_abs.iloc[i, j]
        if coeficiente > 0.85:
            parejas_redundantes.append({
                'Variable 1': matriz_corr_abs.columns[i],
                'Variable 2': matriz_corr_abs.columns[j],
                'Correlación': round(coeficiente, 4)
            })

# 3. Convertir a DataFrame y mostrar ordenado
df_pares = pd.DataFrame(parejas_redundantes).sort_values(by='Correlación', ascending=False)
print("--- PAREJAS ALTAMENTE CORRELACIONADAS (> 0.85) ---")
print(df_pares.to_string(index=False))
print(df.info())
print(df.head())
# %%
