import pandas as pd
import os
from pathlib import Path

def merge_interim_with_depression():
    # 1. Cargar la fuente de verdad: Score de Depresión
    dep_path = "data/processed/dpq_limpio.csv"
    if not os.path.exists(dep_path):
        print(f"❌ Error: No se encuentra {dep_path}. Ejecuta primero la limpieza de DPQ.")
        return

    print(f"Cargando score de depresión desde {dep_path}...")
    df_base = pd.read_csv(dep_path)
    
    # Asegurar que SEQN sea el índice para el merge inicial
    if 'SEQN' not in df_base.columns:
        print("❌ Error: El archivo de depresión no tiene la columna SEQN.")
        return

    # 2. Listar archivos en interim (excluyendo el dpq original para no duplicar esfuerzo)
    interim_dir = Path("data/interim")
    interim_files = [f for f in interim_dir.glob("*.csv") if "dpq_h_columnas" not in f.name]
    
    print(f"Se encontraron {len(interim_files)} archivos en interim para unir.")

    merged_df = df_base.copy()

    # 3. Join iterativo (Inner Join)
    for file_path in interim_files:
        print(f"Uniendo {file_path.name}...")
        df_temp = pd.read_csv(file_path)
        
        # Renombrar ID a SEQN si es necesario (por si acaso hay inconsistencias)
        if 'ID' in df_temp.columns and 'SEQN' not in df_temp.columns:
            df_temp = df_temp.rename(columns={'ID': 'SEQN'})
        
        if 'SEQN' not in df_temp.columns:
            print(f"⚠️ Saltando {file_path.name}: No tiene columna SEQN.")
            continue

        # Seleccionar solo columnas que no estén ya en el merge (excepto SEQN)
        cols_to_use = df_temp.columns.difference(merged_df.columns).tolist()
        if 'SEQN' not in cols_to_use:
            cols_to_use.append('SEQN')

        # Realizamos INNER JOIN para cumplir con la condición: "si no está en score depresión no lo incluyas"
        # Dado que merged_df empieza con los datos de depresión, el inner join garantiza la exclusión.
        merged_df = pd.merge(merged_df, df_temp[cols_to_use], on='SEQN', how='inner')

    # 4. Guardar resultado
    output_path = "data/interim_con_depresion.csv"
    merged_df.to_csv(output_path, index=False)
    
    print("-" * 30)
    print(f"✅ Proceso completado.")
    print(f"Filas finales: {len(merged_df)}")
    print(f"Columnas totales: {len(merged_df.columns)}")
    print(f"Archivo guardado en: {output_path}")

if __name__ == "__main__":
    merge_interim_with_depression()
