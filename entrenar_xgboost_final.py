# %% [1] Importación y Carga
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

print("Cargando dataset...")
df = pd.read_csv('data/interim_con_depresion.csv')

# %% [2] Limpieza de Códigos Especiales (según Notebook 09)
# Reemplazar valores de "No sabe" o "Saltos" por -1.0 en columnas de frecuencia
for col in ['DBD895', 'DBD900']:
    if col in df.columns:
        df[col] = df[col].replace([5555.0, 9999.0], -1.0)

# %% [3] Identificación y Conversión de Categorías
# Mapeo basado en el Notebook 09 y códigos NHANES
cols_a_categoricas = [
    'RIDRETH3', 'DMDEDUC2', 'DMDMARTL', 'RIAGENDR',  # Demográficas
    'DBQ700', 'DRQSDIET',                           # Dieta
    'DUQ200', 'DUQ240', 'DUQ370', 'DUQ430',         # Drogas
    'OCD150', 'OCD231', 'OCD241',                   # Ocupación
    'OCD390G', 'OCD391', 'OCD392', 'OCQ210', 'OCQ260', 
    'SMQ020', 'SMQ040', 'SMQ078', 'SMQ670',         # Tabaco
    'SLQ050', 'SLQ060'                              # Sueño
]

# Filtrar solo las que existen en este dataframe
cols_existentes = [c for c in cols_a_categoricas if c in df.columns]

print(f"Convirtiendo {len(cols_existentes)} variables a categóricas (strings)...")
for col in cols_existentes:
    # Convertimos a string para que get_dummies las trate como categorías
    # Usamos .astype(str) pero manejamos los nulos para que no se vuelvan la palabra "nan"
    df[col] = df[col].apply(lambda x: str(int(x)) if pd.notnull(x) and x != '' and x != 'nan' else 'nan')

# %% [4] Creación de Dummies (One-Hot Encoding)
# Quitamos SEQN antes de los dummies para no crear mil columnas de ID
target = 'score_depresion'
df_ml = df.drop(columns=['SEQN'])

# Crear dummies
df_final = pd.get_dummies(df_ml, columns=cols_existentes, drop_first=True)
print(f"Dimensiones finales del dataset tras One-Hot Encoding: {df_final.shape}")

# %% [5] Entrenamiento del Modelo
X = df_final.drop(columns=[target])
y = df_final[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nEntrenando XGBoost con arquitectura de categorías completa...")
# Añadir configuración para datos con nulos o categorías
model = xgb.XGBRegressor(
    n_estimators=1000, 
    learning_rate=0.03, 
    max_depth=6, 
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    device='cpu' # Aseguramos CPU para compatibilidad
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# %% [6] Evaluación Clínica
y_pred = model.predict(X_test)

# Métricas de Regresión
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Métricas de Clasificación (Riesgo PHQ-9 >= 10)
y_test_bin = (y_test >= 10).astype(int)
y_pred_bin = (y_pred >= 10).astype(int)

print("\n" + "="*50)
print("RESULTADOS FINALES DEL MODELO")
print("="*50)
print(f"R2 Score: {r2:.4f}")
print(f"MAE Global: {mae:.4f}")

print("\n--- DESEMPEÑO CLÍNICO (Detección de Riesgo >= 10) ---")
print(classification_report(y_test_bin, y_pred_bin, target_names=['Sano/Leve', 'Riesgo Clínico']))

# Error por gravedad
resultados = pd.DataFrame({'Real': y_test, 'Pred': y_pred})
resultados['Error'] = np.abs(resultados['Real'] - resultados['Pred'])
resultados['Grupo'] = pd.cut(resultados['Real'], bins=[-1, 9, 27], labels=['Sano/Leve', 'Moderado/Severo'])
error_grupo = resultados.groupby('Grupo', observed=True)['Error'].mean()

print("\n--- MAE POR GRUPO ---")
print(error_grupo.to_string())

# %% [7] Importancia de las nuevas categorías
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\n--- TOP 10 PREDICTORES (Incluyendo Categorías) ---")
print(importances.sort_values(ascending=False).head(10))
