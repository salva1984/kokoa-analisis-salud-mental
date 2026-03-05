import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# 1. Cargar el dataset
print("Cargando datos...")
df = pd.read_csv('data/interim_con_depresion.csv')

# 2. Preprocesamiento básico
# El objetivo es score_depresion
target = 'score_depresion'

# SEQN es solo un ID, no aporta información predictiva
features = [col for col in df.columns if col not in [target, 'SEQN']]

X = df[features]
y = df[target]

# 3. Dividir en entrenamiento y prueba (80/20)
# Corregido: test_size en lugar de test_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset de entrenamiento: {X_train.shape}")
print(f"Dataset de prueba: {X_test.shape}")

# 4. Configurar y entrenar el modelo XGBoost Regressor
print("\nEntrenando modelo XGBoost...")
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    early_stopping_rounds=50
)

# Entrenar con validación temprana para evitar sobreajuste
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# 5. Evaluación del modelo
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Resultados del Modelo ---")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# 6. Importancia de las variables (Top 15)
importances = pd.Series(model.feature_importances_, index=features)
top_15 = importances.sort_values(ascending=False).head(15)

print("\n--- Top 15 Variables más Influyentes ---")
print(top_15)

# Opcional: Graficar importancia de variables
plt.figure(figsize=(10, 6))
top_15.plot(kind='barh')
plt.title('Top 15 Variables más Influyentes en el Score de Depresión')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nGráfico de importancia guardado como 'feature_importance.png'")
