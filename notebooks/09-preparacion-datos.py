# %%
import pandas as pd

def detectar_categorias_ocultas(df, limite_unicos=25):
    print("-" * 50)
    print(f"Buscando columnas numéricas con {limite_unicos} o menos valores distintos...\n")
    print("-" * 50)
    
    columnas_sospechosas = []
    
    for col in df.columns:
        # 1. Solo analizamos las columnas que son numéricas (int o float)
        if pd.api.types.is_numeric_dtype(df[col]):
            valores_unicos = df[col].dropna().unique()
            num_unicos = len(valores_unicos)
            
            # 2. Si tiene pocos valores únicos y no es binaria pura (0 y 1), la mostramos
            if 2 < num_unicos <= limite_unicos:
                columnas_sospechosas.append(col)
                # Ordenamos los valores para que sea más fácil leerlos
                valores_ordenados = sorted(valores_unicos)
                
                print(f"🔹 Columna: {col}")
                print(f"   Tipo de dato: {df[col].dtype}")
                print(f"   Cantidad de valores únicos: {num_unicos}")
                print(f"   Valores: {valores_ordenados}")
                print("-" * 50)
                
    print(f"\n✅ Análisis completado. Se encontraron {len(columnas_sospechosas)} posibles categorías.")
    return columnas_sospechosas

df = pd.read_csv("../data/processed/dataset_limpio.csv")
posibles_cat = detectar_categorias_ocultas(df)
# %%
def auditar_columnas_texto(df):
    print("📝 INICIANDO AUDITORÍA DE COLUMNAS DE TEXTO (OBJECT) 📝")
    print("-" * 50)
    
    # Filtramos automáticamente solo las columnas de texto
    cols_texto = df.select_dtypes(include=['object']).columns
    
    if len(cols_texto) == 0:
        print("No hay columnas tipo object en este DataFrame.")
        return

    for col in cols_texto:
        valores_unicos = df[col].dropna().unique()
        num_unicos = len(valores_unicos)
        
        print(f"🔹 Columna: {col}")
        print(f"   Cantidad de categorías únicas: {num_unicos}")
        
        # Si tiene pocas categorías, mostramos todas para verificar limpieza
        if num_unicos <= 15:
            print(f"   Valores: {valores_unicos}")
        # Si tiene muchas, lanzamos una alerta y mostramos solo una muestra
        else:
            print(f"   ⚠️ ALERTA DE CARDINALIDAD ALTA ⚠️")
            print(f"   Mostrando 5 ejemplos: {valores_unicos[:5]}")
            
        print("-" * 50)
            
    print(f"\n✅ Auditoría completada. Se revisaron {len(cols_texto)} columnas de texto.")

# Ejecutamos la función sobre el dataset
auditar_columnas_texto(df)
# %%
df = df.copy()
for col in ['DBD895', 'DBD900']:
    if col in df.columns:
        df[col] = df[col].replace([5555.0, 9999.0], -1.0)

cols_a_texto = [
    'Etnia', 'Nivel_Educativo', 'Estado_Civil', 'DBQ700', 'DRQSDIET', 
    'DUQ200', 'DUQ240', 'DUQ370', 'DUQ430', 'OCD150', 'OCD231', 'OCD241', 
    'OCD390G', 'OCD391', 'OCD392', 'OCQ210', 'OCQ260', 'SMQ020', 'SMQ040', 
    'SMQ078', 'SMQ670','SLQ50', 'SLQ060'
]

# 4. Convertimos a string
for col in cols_a_texto:
    if col in df.columns:
        df[col] = df[col].astype(str)

# 5. La magia de los Dummies (Transforma todo el texto a 0s y 1s)
df = pd.get_dummies(df, drop_first=True)
df.to_csv("../data/processed/dataset_final_dummies.csv", index=False)
print(f"Data lista para el modelo. Dimensiones de df: {df.shape}")
# %%
df.info()
#
# %%
