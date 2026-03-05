# %% [1] Importación de librerías y carga de datos
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

print("Cargando y preparando datos...")
df = pd.read_csv('data/interim_con_depresion.csv')

# Definir Target y Features
target = 'score_depresion'
features = [col for col in df.columns if col not in [target, 'SEQN']]

X = df[features]
y = df[target]

# %% [2] Entrenamiento del modelo (XGBoost)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Entrenando XGBoost...")
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# %% [3] Generar predicciones
y_pred = model.predict(X_test)

# Creamos un DataFrame para comparar resultados
resultados = pd.DataFrame({
    'ID_Sujeto': df.loc[X_test.index, 'SEQN'].astype(int),
    'Score_Real': y_test.values,
    'Score_Predicho': np.round(y_pred, 2),
    'Error_Absoluto': np.round(np.abs(y_test.values - y_pred), 2)
})

# %% [4] Ver predicciones en crudo (20 casos aleatorios)
print("\n--- MUESTRA DE PREDICCIONES (Realidad vs Modelo) ---")
# Mezclamos para ver casos variados
print(resultados.sample(20, random_state=7).to_string(index=False))

# %% [5] Análisis de precisión por rango
print("\n--- PRECISIÓN GLOBAL ---")
print(f"Error promedio global (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

# %% [6] Visualización: Gráfico de Dispersión
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred, alpha=0.4, color='teal', edgecolors='k')
plt.plot([0, 27], [0, 27], color='red', linestyle='--', label='Predicción Perfecta (1:1)')
plt.title('Dispersión de Predicciones: Score Real vs Score Predicho', fontsize=14)
plt.xlabel('Score PHQ-9 Real (Realidad)', fontsize=12)
plt.ylabel('Score PHQ-9 Predicho (Modelo)', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('visualizacion_predicciones.png')

print("\nGráfico visual guardado como 'visualizacion_predicciones.png'")
