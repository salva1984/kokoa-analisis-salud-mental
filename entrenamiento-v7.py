# %% [1] Preparación de Datos y Modelo
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve, 
    average_precision_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

print("Iniciando Entrenamiento V7 (Clasificador)...")
df = pd.read_csv('data/interim_con_depresion.csv')

# Listas de variables (Mixed Truth)
cols_categoricas = ['RIAGENDR', 'RIDRETH3', 'DMDEDUC2', 'DMDEDUC3', 'DMDMARTL', 'DMQMILIZ', 'DRQSDIET', 'DBQ700', 'ALQ101', 'ALQ110', 'DUQ200', 'DUQ240', 'DUQ370', 'DUQ430', 'DLQ010', 'DLQ020', 'DLQ040', 'DLQ050', 'DLQ060', 'DLQ080', 'SMQ020', 'SMQ040', 'SMQ078', 'SMQ670', 'SLQ050', 'SLQ060', 'MCQ160M', 'MCQ160L', 'MCQ160A', 'MCQ160F', 'MCQ160E', 'MCQ160C', 'MCQ220', 'MCQ010', 'MCQ080', 'MCQ070', 'OCD150', 'OCQ210', 'OCD231', 'OCD241', 'OCQ260', 'OCD390G', 'OCD391', 'OCD392', 'PAQ605', 'PAQ620', 'PAQ635', 'PAQ650', 'PAQ665', 'DRQSDT1', 'DRQSDT10', 'DRQSDT11', 'DRQSDT12', 'DRQSDT2', 'DRQSDT3', 'DRQSDT4', 'DRQSDT5', 'DRQSDT6', 'DRQSDT7', 'DRQSDT8', 'DRQSDT9', 'DRQSDT91', 'WHD080E', 'WHD080K', 'WHD080P', 'WHQ030', 'WHQ060', 'WHQ070', 'SMD630', 'SMQ621']
cols_cuantitativas = ['RIDAGEYR', 'INDFMPIR', 'BMXBMI', 'BMXWAIST', 'ALQ120_dias_anio', 'ALQ130', 'ALQ141_dias_anio', 'ALQ160', 'DBD895', 'DBD900', 'DBD905', 'DBD910', 'DUQ230', 'DUQ210', 'DUQ280', 'DUQ320', 'DUQ360', 'DUQ410', 'SMD641', 'SMD650', 'SMD030', 'SMQ848', 'SMQ852_dias_total', 'SLD010H', 'OCQ180', 'OCD395', 'PAD680', 'OCD270', 'WHD010', 'WHD020']

# Limpieza y Encoding
for col in [c for c in cols_cuantitativas if c in df.columns]:
    df[col] = df[col].replace([77, 99, 777, 999, 7777, 9999, 5555, 77777, 99999], np.nan)
for col in [c for c in cols_categoricas if c in df.columns]:
    df[col] = df[col].apply(lambda x: str(int(x)) if pd.notnull(x) and x != '' and x != 'nan' else 'nan')

df['target_bin'] = (df['score_depresion'] >= 10).astype(int)
existentes = [c for c in (cols_cuantitativas + cols_categoricas) if c in df.columns]
df_final = pd.get_dummies(df[existentes + ['target_bin']], columns=[c for c in cols_categoricas if c in df.columns], drop_first=True)

X = df_final.drop(columns=['target_bin'])
y = df_final['target_bin']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Balanceo y Entrenamiento
ratio = len(y_train[y_train==0]) / len(y_train[y_train==1])
clf = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4, scale_pos_weight=ratio, eval_metric='aucpr', early_stopping_rounds=50, random_state=42)
clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# %% [2] Análisis de Umbral y Curva ROC (User Logic)
y_probs = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Buscar umbral para TPR >= 0.85
idx = np.where(tpr >= 0.85)[0][0]
umbral_optimo = thresholds[idx]

print(f"🎯 Umbral encontrado (85% Recall): {umbral_optimo:.4f}")
print(f"✅ Sensitivity (Recall): {tpr[idx]:.2%}")
print(f"⚠️ FPR: {fpr[idx]:.2%}")

# %% [3] Reporte Final con Umbral Ajustable
umbral_final = 0.6
y_pred_final = (y_probs >= umbral_final).astype(int)
print("\n--- REPORTE FINAL PRIORIZANDO SENSITIVITY ---")
print(classification_report(y_test, y_pred_final))

# %% [4] Gráfico ROC (Cuadrado)
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc_score(y_test, y_probs):.2f}', color='darkorange')
plt.plot([0, 1], [0, 1], 'k--', label='Azar (0.5)')
plt.scatter(fpr[idx], tpr[idx], color='red', label=f'Punto Seleccionado (T={umbral_final:.2f})')
plt.xlabel('Tasa de Falsos Positivos'); plt.ylabel('Tasa de Verdaderos Positivos'); plt.title('Curva ROC')
plt.axis('square'); plt.legend(); plt.grid(alpha=0.3)
plt.savefig('v7_roc.png')

# %% [5] Gráfico Precision-Recall (Cuadrado)
plt.figure(figsize=(8, 8))
p, r, _ = precision_recall_curve(y_test, y_probs)
plt.plot(r, p, color='deeppink', label=f'AP = {average_precision_score(y_test, y_probs):.2f}')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Curva Precision-Recall')
plt.axis('square'); plt.legend(); plt.grid(alpha=0.3)
plt.savefig('v7_pr.png')

# %% [6] Matriz de Confusión
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_final), annot=True, fmt='d', cmap='RdPu')
plt.title(f'Matriz de Confusion (T={umbral_final:.2f})')
plt.xlabel('Prediccion'); plt.ylabel('Realidad')
plt.savefig('v7_confusion.png')

# %%
