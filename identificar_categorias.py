import pandas as pd
import numpy as np

def detectar_categorias_ocultas(df, limite_unicos=25):
    print("-" * 50)
    print(f"Buscando columnas numéricas con {limite_unicos} o menos valores distintos...\n")
    print("-" * 50)
    
    columnas_sospechosas = []
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            valores_unicos = df[col].dropna().unique()
            num_unicos = len(valores_unicos)
            
            # Si tiene pocos valores únicos y no es binaria pura (0 y 1), o es binaria con códigos tipo NHANES (1, 2)
            if 1 < num_unicos <= limite_unicos:
                columnas_sospechosas.append(col)
                valores_ordenados = sorted(valores_unicos)
                
                print(f"🔹 Columna: {col}")
                print(f"   Cantidad de valores únicos: {num_unicos}")
                # Formatear valores para que no salgan científicos
                valores_format = [round(v, 2) if isinstance(v, float) else v for v in valores_ordenados]
                print(f"   Valores: {valores_format}")
                print("-" * 50)
                
    print(f"\n✅ Análisis completado. Se encontraron {len(columnas_sospechosas)} posibles categorías.")
    return columnas_sospechosas

# Cargar el dataset actual
df = pd.read_csv('data/interim_con_depresion.csv')

# Ejecutar detección
posibles_cat = detectar_categorias_ocultas(df)
