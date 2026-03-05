# %% [1] Importación y Carga
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

print("Cargando dataset...")
df = pd.read_csv('data/interim_con_depresion.csv')

# %% [2] Preparación de Categorías (One-Hot Encoding)
cols_a_categoricas = [
    'RIDRETH3', 'DMDEDUC2', 'DMDMARTL', 'RIAGENDR', 
    'DBQ700', 'DRQSDIET', 'DUQ200', 'DUQ240', 'DUQ370', 'DUQ430',
    'OCD150', 'OCD231', 'OCD241', 'OCD390G', 'OCD391', 'OCD392', 
    'OCQ210', 'OCQ260', 'SMQ020', 'SMQ040', 'SMQ078', 'SMQ670',
    'SLQ050', 'SLQ060'
]

cols_existentes = [c for c in cols_a_categoricas if c in df.columns]

for col in cols_existentes:
    df[col] = df[col].apply(lambda x: str(int(x)) if pd.notnull(x) and x != '' and x != 'nan' else 'nan')

target = 'score_depresion'
df_ml = df.drop(columns=['SEQN'])
df_final = pd.get_dummies(df_ml, columns=cols_existentes, drop_first=True)

X = df_final.drop(columns=[target])
y = df_final[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [3] BALANCEO DE CLASES (Weighted Regression)
# Creamos un vector de pesos: los casos de depresión (>=10) pesan 5 veces más
weights_train = np.where(y_train >= 10, 5.0, 1.0)

# %% [4] Configuración de Hiperparámetros
# Puedes ajustar estos valores para ver cómo cambia el R2 y el Recall
params = {
    'n_estimators': 1000,
    'learning_rate': 0.02,    # Paso más pequeño para mayor precisión
    'max_depth': 5,           # Un poco menos profundo para evitar overfitting
    'subsample': 0.8,
    'colsample_bytree': 0.7,  # Usamos menos variables por árbol para diversificar
    'random_state': 42,
    'reg_alpha': 0.1,         # Regularización L1
    'reg_lambda': 1.0,        # Regularización L2
    'n_jobs': -1
}

print("
Entrenando Modelo entrenamiento-v5 (Weighted XGBoost)...")
model = xgb.XGBRegressor(**params, early_stopping_rounds=50)

model.fit(
    X_train, y_train,
    sample_weight=weights_train, # APLICAMOS LOS PESOS AQUÍ
    eval_set=[(X_test, y_test)],
    verbose=False
)

# %% [5] Evaluación Integral
y_pred = model.predict(X_test)
y_test_bin = (y_test >= 10).astype(int)
y_pred_bin = (y_pred >= 10).astype(int)

print("
" + "="*50)
print("RESULTADOS ENTRENAMIENTO-V5 (PESOS AJUSTADOS)")
print("="*50)
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE Global: {mean_absolute_error(y_test, y_pred):.4f}")

print("
--- DESEMPEÑO CLÍNICO (Detección de Riesgo >= 10) ---")
# Observa si el Recall de 'Riesgo Clínico' subió
print(classification_report(y_test_bin, y_pred_bin, target_names=['Sano/Leve', 'Riesgo Clínico']))

# Error por gravedad
resultados = pd.DataFrame({'Real': y_test, 'Pred': y_pred})
resultados['Error'] = np.abs(resultados['Real'] - resultados['Pred'])
resultados['Grupo'] = pd.cut(resultados['Real'], bins=[-1, 9, 27], labels=['Sano/Leve', 'Moderado/Severo'])
print("
--- MAE POR GRUPO ---")
print(resultados.groupby('Grupo', observed=True)['Error'].mean().to_string())
