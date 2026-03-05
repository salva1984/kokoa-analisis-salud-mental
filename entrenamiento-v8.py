# %% [1] Importación y Carga de Datos
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve, 
    average_precision_score, roc_auc_score, roc_curve, f1_score, make_scorer
)
import matplotlib.pyplot as plt
import seaborn as sns

print("Cargando dataset para entrenamiento-v8 (Optuna HPO)...")
df = pd.read_csv('data/interim_con_depresion.csv')

# %% [2] Preprocesamiento y Clasificación de Variables
cols_categoricas = ['RIAGENDR', 'RIDRETH3', 'DMDEDUC2', 'DMDEDUC3', 'DMDMARTL', 'DMQMILIZ', 'DRQSDIET', 'DBQ700', 'ALQ101', 'ALQ110', 'DUQ200', 'DUQ240', 'DUQ370', 'DUQ430', 'DLQ010', 'DLQ020', 'DLQ040', 'DLQ050', 'DLQ060', 'DLQ080', 'SMQ020', 'SMQ040', 'SMQ078', 'SMQ670', 'SLQ050', 'SLQ060', 'MCQ160M', 'MCQ160L', 'MCQ160A', 'MCQ160F', 'MCQ160E', 'MCQ160C', 'MCQ220', 'MCQ010', 'MCQ080', 'MCQ070', 'OCD150', 'OCQ210', 'OCD231', 'OCD241', 'OCQ260', 'OCD390G', 'OCD391', 'OCD392', 'PAQ605', 'PAQ620', 'PAQ635', 'PAQ650', 'PAQ665', 'DRQSDT1', 'DRQSDT10', 'DRQSDT11', 'DRQSDT12', 'DRQSDT2', 'DRQSDT3', 'DRQSDT4', 'DRQSDT5', 'DRQSDT6', 'DRQSDT7', 'DRQSDT8', 'DRQSDT9', 'DRQSDT91', 'WHD080E', 'WHD080K', 'WHD080P', 'WHQ030', 'WHQ060', 'WHQ070', 'SMD630', 'SMQ621']
cols_cuantitativas = ['RIDAGEYR', 'INDFMPIR', 'BMXBMI', 'BMXWAIST', 'ALQ120_dias_anio', 'ALQ130', 'ALQ141_dias_anio', 'ALQ160', 'DBD895', 'DBD900', 'DBD905', 'DBD910', 'DUQ230', 'DUQ210', 'DUQ280', 'DUQ320', 'DUQ360', 'DUQ410', 'SMD641', 'SMD650', 'SMD030', 'SMQ848', 'SMQ852_dias_total', 'SLD010H', 'OCQ180', 'OCD395', 'PAD680', 'OCD270', 'WHD010', 'WHD020']

# Limpieza y Encoding
for col in [c for c in cols_cuantitativas if c in df.columns]:
    df[col] = df[col].replace([77, 99, 777, 999, 7777, 9999, 5555, 77777, 99999], np.nan)
for col in [c for c in cols_categoricas if c in df.columns]:
    df[col] = df[col].apply(lambda x: str(int(x)) if pd.notnull(x) and x != '' and x != 'nan' else 'nan')

df['target_riesgo'] = (df['score_depresion'] >= 10).astype(int)
existentes = [c for c in (cols_cuantitativas + cols_categoricas) if c in df.columns]
df_final = pd.get_dummies(df[existentes + ['target_riesgo']], columns=[c for c in cols_categoricas if c in df.columns], drop_first=True)

X = df_final.drop(columns=['target_riesgo'])
y = df_final['target_riesgo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %% [3] Optimización de Hiperparámetros con Optuna (Target: F1-Score)
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 20),
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    }
    
    model = xgb.XGBClassifier(**params)
    # Usamos cross-validation para optimizar el F1-score
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='f1').mean()
    return score

print("Iniciando búsqueda con Optuna...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print(f"Mejores parámetros: {study.best_params}")
print(f"Mejor F1-Score (CV): {study.best_value:.4f}")

# %% [4] Entrenamiento Final con Mejores Parámetros
best_params = study.best_params
clf = xgb.XGBClassifier(**best_params, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# %% [5] Análisis de Umbral y Resultados
y_probs = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds_roc = roc_curve(y_test, y_probs)

# Buscamos el umbral que maximice el F1-score en el set de prueba (o puedes usar el 0.5 por defecto)
precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-9)
idx_f1 = np.argmax(f1_scores)
umbral_f1 = thresholds_pr[min(idx_f1, len(thresholds_pr)-1)]

print(f"🎯 Umbral para máximo F1-score: {umbral_f1:.4f}")

# %% [6] Reporte Clínico (Umbral Ajustable)
# --- CAMBIA ESTE VALOR PARA EXPERIMENTAR ---
umbral_final = 0.3
# -------------------------------------------

y_pred_final = (y_probs >= umbral_final).astype(int)

print("" + "="*50)
print(f"REPORTE FINAL V8 (UMBRAL F1: {umbral_final:.4f})")
print("="*50)
print(classification_report(y_test, y_pred_final, target_names=['Sano/Leve', 'Riesgo Clinico']))

# %% [7] Gráficos Separados (Cuadrados)
# 1. ROC Curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc_score(y_test, y_probs):.2f}', color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.scatter(fpr[np.argmin(np.abs(thresholds_roc - umbral_final))], tpr[np.argmin(np.abs(thresholds_roc - umbral_final))], color='red', label=f'Umbral {umbral_final:.2f}')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Curva ROC (v8)')
plt.axis('square'); plt.legend(); plt.grid(alpha=0.3)
plt.savefig('v8_roc.png')

# 2. PR Curve
plt.figure(figsize=(8, 8))
plt.plot(recall_vals, precision_vals, color='deeppink', lw=2, label=f'AP = {average_precision_score(y_test, y_probs):.2f}')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Curva Precision-Recall (v8)')
plt.axis('square'); plt.legend(); plt.grid(alpha=0.3)
plt.savefig('v8_pr.png')

# 3. Matriz de Confusión
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_final), annot=True, fmt='d', cmap='YlGnBu')
plt.title(f'Matriz de Confusión (T={umbral_final:.2f})')
plt.xlabel('Predicción'); plt.ylabel('Realidad')
plt.savefig('v8_confusion.png')

# %% [8] Importancia de Variables
importances = pd.Series(clf.feature_importances_, index=X.columns)
print("--- TOP 10 FACTORES DETERMINANTES (v8) ---")
print(importances.sort_values(ascending=False).head(10))


# Mejores parámetros: {'n_estimators': 208, 'max_depth': 4, 'learning_rate': 0.028121605189448867, 'subsample': 0.5205630141774021, 'colsample_bytree': 0.7671667438316181, 'scale_pos_weight': 4.930935206714419}
# Mejor F1-Score (CV): 0.4056