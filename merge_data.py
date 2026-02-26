import pandas as pd
import os

def merge_cleaned_datasets():
    files = [
        "data/demo.csv",
        "data/alq.csv",
        "data/mcq.csv",
        "data/paq.csv",
        "data/whq.csv",
        "data/processed/bmx_limpio.csv",
        "data/processed/dbq_limpio.csv",
        "data/processed/dr1tot_limpio.csv",
        "data/processed/duq_limpio.csv",
        "data/limpieza_G3/dlq_limpio.csv",
        "data/limpieza_G3/ocq_limpio.csv",
        "data/limpieza_G3/slq_limpio.csv",
        "data/limpieza_G3/smq_limpio.csv"
    ]
    
    existing_files = []
    
    # Nuevo bloque de validación individual
    for f in files:
        if os.path.exists(f):
            existing_files.append(f)
        else:
            print(f"⚠️ Advertencia: Archivo no encontrado -> {f}")
            
    print(f"\nArchivos encontrados para unir: {len(existing_files)}")
    
    # Seguro adicional por si no hay ningún archivo
    if not existing_files:
        print("Error: No hay archivos disponibles para el merge. Terminando ejecución.")
        return

    merged_df = None
    
    for file_path in existing_files:
        df = pd.read_csv(file_path)
        
        # Mantenemos las columnas originales. Solo renombramos 'ID' si es estrictamente necesario para el merge.
        if 'ID' in df.columns and 'SEQN' not in df.columns:
            df = df.rename(columns={'ID': 'SEQN'})
        
        if merged_df is None:
            merged_df = df
        else:
            # Seleccionamos solo las columnas nuevas para evitar duplicados, manteniendo SEQN para el cruce.
            cols_to_use = df.columns.difference(merged_df.columns).tolist()
            if 'SEQN' not in cols_to_use:
                cols_to_use.append('SEQN')
                
            merged_df = pd.merge(merged_df, df[cols_to_use], on='SEQN', how='outer')
            
    # 1. Limpieza de SEQN
    merged_df = merged_df.dropna(subset=['SEQN'])
    
    # 2. FILTRO DE EDAD: Solo mayores de 18 años
    col_edad = next((c for c in merged_df.columns if c.upper() == 'EDAD'), None)
    
    if col_edad:
        antes = len(merged_df)
        merged_df = merged_df.loc[merged_df[col_edad] >= 18]
        despues = len(merged_df)
        print(f"Filtro aplicado ({col_edad} >= 18): Se eliminaron {antes - despues} registros.")
    else:
        print("Advertencia: No se encontró la columna de edad para filtrar.")

    # Ordenar por SEQN y guardar
    merged_df = merged_df.sort_values('SEQN')
    print(f"Dataset final unido: {merged_df.shape}")
    
    output_path = "data/dataset_completo.csv"
    
    # Aseguramos que la carpeta de destino exista antes de guardar
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    merged_df.to_csv(output_path, index=False)
    print(f"Dataset guardado en {output_path}")

if __name__ == "__main__":
    merge_cleaned_datasets()