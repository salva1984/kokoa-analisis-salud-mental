# %% [1] Importación y Carga
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

print("Cargando dataset para entrenamiento-v6...")
df = pd.read_csv('data/interim_con_depresion.csv')

# %% [2] Mapeo de la "Verdad Mixta" (Clínica + Estadística)

# 1. Variables Categóricas (Cualitativas -> One-Hot Encoding)
cols_categoricas = [
    'RIAGENDR', 'RIDRETH3', 'DMDEDUC2', 'DMDMARTL', 'DMQMILIZ', # Demografía
    'DRQSDIET', 'DBQ700',                                      # Dieta/Salud
    'ALQ101', 'ALQ110',                                        # Alcohol Status
    'DUQ200', 'DUQ240', 'DUQ370', 'DUQ430',                    # Drogas Status
    'DLQ010', 'DLQ020', 'DLQ040', 'DLQ050', 'DLQ060', 'DLQ080', # Discapacidades
    'SMQ020', 'SMQ040', 'SMQ078', 'SMQ670',                    # Tabaco Status/Ordinal
    'SLQ050', 'SLQ060',                                        # Sueño Status
    'MCQ160M', 'MCQ160L', 'MCQ160A', 'MCQ160F', 'MCQ160E', 
    'MCQ160C', 'MCQ220', 'MCQ010', 'MCQ080', 'MCQ070',         # Condiciones Médicas
    'OCQ210', 'OCD231', 'OCD241', 'OCQ260', 'OCD390G', 
    'OCD391', 'OCD392',                                        # Ocupación
    'PAQ605', 'PAQ620', 'PAQ635', 'PAQ650', 'PAQ665'           # Actividad Física
]

# 2. Variables Cuantitativas (Escala -> No Encoding)
cols_cuantitativas = [
    'RIDAGEYR', 'INDFMPIR', 'BMXBMI', 'BMXWAIST',
    'ALQ120_dias_anio', 'ALQ130', 'ALQ141_dias_anio', 'ALQ160',
    'DBD895', 'DBD900', 'DBD905', 'DBD910',
    'DUQ230', 'DUQ210', 'DUQ280', 'DUQ320', 'DUQ360', 'DUQ410',
    'SMD641', 'SMD650', 'SMD030', 'SMQ848', 'SMQ852_dias_total',
    'SLD010H', 'OCQ180', 'OCD395', 'PAD680'
]

# %% [3] Limpieza y Preprocesamiento

existentes_cat = [c for c in cols_categoricas if c in df.columns]
existentes_num = [c for c in cols_cuantitativas if c in df.columns]

# Limpiar códigos de error en variables cuantitativas
for col in existentes_num:
    df[col] = df[col].replace([77, 99, 777, 999, 7777, 9999, 5555, 77777, 99999], np.nan)

# Convertir categóricas a string
for col in existentes_cat:
    df[col] = df[col].apply(lambda x: str(int(x)) if pd.notnull(x) and x != '' and x != 'nan' else 'nan')

# Crear el dataset final para ML
target = 'score_depresion'
df_ml = df[existentes_num + existentes_cat + [target]].copy()
df_final = pd.get_dummies(df_ml, columns=existentes_cat, drop_first=True)

# %% [4] División y Pesos
X = df_final.drop(columns=[target])
y = df_final[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pesos de muestra para balanceo clínico (PHQ-9 >= 10)
weights_train = np.where(y_train >= 10, 5.0, 1.0)

# %% [5] Búsqueda de Hiperparámetros (Grid Search)
print("\nIniciando búsqueda de mejores hiperparámetros (Grid Search)...")
param_grid = {
    'max_depth': [4, 6, 8, None],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.7, 0.9]
}

# Usamos un modelo base para el Grid Search
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=1
)

grid_search.fit(X_train, y_train, sample_weight=weights_train)

best_params = grid_search.best_params_
print(f"\nMejores parámetros encontrados: {best_params}")

# %% [6] Entrenamiento Final con Mejores Parámetros
print("\nEntrenando modelo final entrenamiento-v6 con mejores parámetros...")
final_model = xgb.XGBRegressor(
    n_estimators=1000,
    **best_params,
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1
)

final_model.fit(
    X_train, y_train,
    sample_weight=weights_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# %% [7] Evaluación Integral
y_pred = final_model.predict(X_test)
y_test_bin = (y_test >= 10).astype(int)
y_pred_bin = (y_pred >= 10).astype(int)

print("\n" + "="*50)
print("RESULTADOS FINALES ENTRENAMIENTO-V6")
print("="*50)
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE Global: {mean_absolute_error(y_test, y_pred):.4f}")

print("\n--- DESEMPEÑO CLÍNICO (Detección de Riesgo >= 10) ---")
print(classification_report(y_test_bin, y_pred_bin, target_names=['Sano/Leve', 'Riesgo Clínico']))

# Análisis de error por grupo
resultados = pd.DataFrame({'Real': y_test, 'Pred': y_pred})
resultados['Error'] = np.abs(resultados['Real'] - resultados['Pred'])
resultados['Grupo'] = pd.cut(resultados['Real'], bins=[-1, 9, 27], labels=['Sano/Leve', 'Moderado/Severo'])

print("\n--- MAE POR GRUPO ---")
print(resultados.groupby('Grupo', observed=True)['Error'].mean().to_string())

# %% [8] Importancia de Variables
importances = pd.Series(final_model.feature_importances_, index=X.columns)
print("\n--- TOP 10 VARIABLES MÁS INFLUYENTES ---")
print(importances.sort_values(ascending=False).head(10))
