# %%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, mean_absolute_error, r2_score, 
    confusion_matrix, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

print("Cargando dataset para entrenamiento-v6.5 (Peso 50 + PR Curve)...")
df = pd.read_csv('data/interim_con_depresion.csv')

# %% [1] Clasificacion de Variables

cols_categoricas = [
    'RIAGENDR', 'RIDRETH3', 'DMDEDUC2', 'DMDEDUC3', 'DMDMARTL', 'DMQMILIZ', 
    'DRQSDIET', 'DBQ700', 'ALQ101', 'ALQ110', 'DUQ200', 'DUQ240', 
    'DUQ370', 'DUQ430', 'DLQ010', 'DLQ020', 'DLQ040', 'DLQ050', 
    'DLQ060', 'DLQ080', 'SMQ020', 'SMQ040', 'SMQ078', 'SMQ670', 
    'SLQ050', 'SLQ060', 'MCQ160M', 'MCQ160L', 'MCQ160A', 'MCQ160F', 
    'MCQ160E', 'MCQ160C', 'MCQ220', 'MCQ010', 'MCQ080', 'MCQ070', 
    'OCD150', 'OCQ210', 'OCD231', 'OCD241', 'OCQ260', 'OCD390G', 
    'OCD391', 'OCD392', 'PAQ605', 'PAQ620', 'PAQ635', 'PAQ650', 'PAQ665',
    'DRQSDT1', 'DRQSDT10', 'DRQSDT11', 'DRQSDT12', 'DRQSDT2', 'DRQSDT3', 
    'DRQSDT4', 'DRQSDT5', 'DRQSDT6', 'DRQSDT7', 'DRQSDT8', 'DRQSDT9', 'DRQSDT91',
    'WHD080E', 'WHD080K', 'WHD080P', 'WHQ030', 'WHQ060', 'WHQ070', 'SMD630', 'SMQ621'
]

cols_cuantitativas = [
    'RIDAGEYR', 'INDFMPIR', 'BMXBMI', 'BMXWAIST',
    'ALQ120_dias_anio', 'ALQ130', 'ALQ141_dias_anio', 'ALQ160',
    'DBD895', 'DBD900', 'DBD905', 'DBD910',
    'DUQ230', 'DUQ210', 'DUQ280', 'DUQ320', 'DUQ360', 'DUQ410',
    'SMD641', 'SMD650', 'SMD030', 'SMQ848', 'SMQ852_dias_total',
    'SLD010H', 'OCQ180', 'OCD395', 'PAD680',
    'OCD270', 'WHD010', 'WHD020'
]

# %% [2] Preprocesamiento

existentes_cat = [c for c in cols_categoricas if c in df.columns]
existentes_num = [c for c in cols_cuantitativas if c in df.columns]

for col in existentes_num:
    df[col] = df[col].replace([77, 99, 777, 999, 7777, 9999, 5555, 77777, 99999], np.nan)

for col in existentes_cat:
    df[col] = df[col].apply(lambda x: str(int(x)) if pd.notnull(x) and x != '' and x != 'nan' else 'nan')

target = 'score_depresion'
df_ml = df[existentes_num + existentes_cat + [target]].copy()
df_final = pd.get_dummies(df_ml, columns=existentes_cat, drop_first=True)

X = df_final.drop(columns=[target])
y = df_final[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# BALANCEO AGRESIVO: Peso 50 para casos de Riesgo (Score >= 10)
weights_train = np.where(y_train >= 10, 50.0, 1.0)

# %% [3] Entrenamiento

print(f"Entrenando modelo con {X.shape[1]} columnas y PESO 50...")
final_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.7,
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

# %% [4] Evaluacion de Deteccion

y_pred = final_model.predict(X_test)
y_test_bin = (y_test >= 10).astype(int)

print("\n" + "="*50)
print("RESULTADOS V6.5 (DETECCION MAXIMIZADA)")
print("="*50)
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE Global: {mean_absolute_error(y_test, y_pred):.4f}")

# Umbral clinico estandar
umbral_score = 10
y_pred_custom = (y_pred >= umbral_score).astype(int)

print(f"\n--- REPORTE CLINICO (SCORE >= {umbral_score}) ---")
print(classification_report(y_test_bin, y_pred_custom, target_names=['Sano/Leve', 'Riesgo Clinico']))

# %% [5] GRAFICO: Curva Precision-Recall
# Usamos y_pred (score continuo) como ranking de severidad
precision, recall, thresholds = precision_recall_curve(y_test_bin, y_pred)
avg_precision = average_precision_score(y_test_bin, y_pred)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR Curve (AP = {avg_precision:.2f})')
plt.fill_between(recall, precision, alpha=0.2, color='orange')
plt.xlabel('Recall (Capacidad de Detección)')
plt.ylabel('Precision (Certeza de la Alerta)')
plt.title('Curva Precision-Recall: Deteccion de Riesgo Clinico')
plt.legend(loc="upper right")
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('precision_recall_v6.5.png')

print("\nCurva Precision-Recall guardada como 'precision_recall_v6.5.png'")

# %% [6] Matriz de Confusion
cm = confusion_matrix(y_test_bin, y_pred_custom)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
            xticklabels=['Sano', 'Riesgo'], yticklabels=['Sano', 'Riesgo'])
plt.title(f'Matriz de Confusion (Peso 50, Umbral: {umbral_score})')
plt.xlabel('Prediccion')
plt.ylabel('Realidad')
plt.savefig('matriz_confusion_v6.5_peso50.png')

print("Matriz de confusion guardada como 'matriz_confusion_v6.5_peso50.png'")

importances = pd.Series(final_model.feature_importances_, index=X.columns)
print("\n--- TOP 10 VARIABLES MAS INFLUYENTES ---")
print(importances.sort_values(ascending=False).head(10))

# %%
