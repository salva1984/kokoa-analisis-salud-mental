# %% [1] Preparación y Entrenamiento (Base)
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
print("Cargando datos para evaluación clínica...")
df = pd.read_csv('data/interim_con_depresion.csv')
target = 'score_depresion'
features = [col for col in df.columns if col not in [target, 'SEQN']]

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Entrenando modelo XGBoost...")
model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Crear DataFrame de resultados para análisis por rangos
resultados = pd.DataFrame({'Score_Real': y_test.values, 'Score_Predicho': y_pred})
resultados['Error_Absoluto'] = np.abs(resultados['Score_Real'] - resultados['Score_Predicho'])

# %% [2] CAMBIO 1: Evaluación Híbrida (Umbral Clínico >= 10)
# Convertir a binario: 1 si es Depresión Moderada/Severa (Riesgo), 0 si es Sano/Leve
y_test_bin = (y_test >= 10).astype(int)
y_pred_bin = (y_pred >= 10).astype(int)

print("\n" + "="*50)
print("1. DESEMPEÑO CLÍNICO (Clasificación Umbral >= 10)")
print("="*50)
print(classification_report(y_test_bin, y_pred_bin, target_names=['Sano/Leve', 'Riesgo Clínico']))

# %% [3] CAMBIO 2: Análisis de Error por Rango de Gravedad
# Categorías oficiales del PHQ-9
resultados['Gravedad_Real'] = pd.cut(
    resultados['Score_Real'], 
    bins=[-1, 4, 9, 14, 19, 27], 
    labels=['Mínima (0-4)', 'Leve (5-9)', 'Moderada (10-14)', 'Mod. Severa (15-19)', 'Severa (20-27)']
)

# Calcular el error promedio por cada nivel
error_por_rango = resultados.groupby('Gravedad_Real', observed=True)['Error_Absoluto'].mean()

print("\n" + "="*50)
print("2. MAE (Error Promedio) POR NIVEL DE GRAVEDAD")
print("="*50)
print(error_por_rango.to_string())

# %% [4] CAMBIO 3: Gráfico de Residuos y Sesgo
plt.figure(figsize=(14, 6))

# Subplot 1: Residuos (¿Subestimamos o sobreestimamos?)
plt.subplot(1, 2, 1)
residuos = y_test - y_pred
plt.scatter(y_pred, residuos, alpha=0.5, color='crimson', edgecolors='white')
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Gráfico de Residuos (Diferencia Real - Predicho)', fontsize=12)
plt.xlabel('Predicción del Modelo (Score)', fontsize=10)
plt.ylabel('Residuo (Error Real)', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)

# Subplot 2: Error por Gravedad (Visualización del sesgo por rango)
plt.subplot(1, 2, 2)
sns.barplot(x=error_por_rango.index, y=error_por_rango.values, hue=error_por_rango.index, palette='viridis', legend=False)
plt.xticks(rotation=45)
plt.title('Error Promedio según Gravedad Real del Paciente', fontsize=12)
plt.ylabel('Error Absoluto Medio (Puntos)', fontsize=10)
plt.xlabel('Nivel de Depresión Real', fontsize=10)

plt.tight_layout()
plt.savefig('analisis_clinico_detallado.png')

print("\n" + "="*50)
print("Análisis visual guardado: 'analisis_clinico_detallado.png'")
print("="*50 + "\n")
