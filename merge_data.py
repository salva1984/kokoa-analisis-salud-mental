import pandas as pd
import glob
import os

def merge_cleaned_datasets():
    # List all cleaned datasets identified
    files = [
        "data/demo.csv",
        "data/alq.csv",
        "data/mcq.csv",
        "data/paq.csv",
        "data/whq.csv",
        "data/limpieza_G2/BMX_H_LIMPIO.csv",
        "data/limpieza_G2/DBQ_H_LIMPIO.csv",
        "data/limpieza_G2/DR1TOT_H_LIMPIO.csv",
        "data/limpieza_G2/DUQ_H_limpio.csv",
        "data/limpieza_G3/dlq_limpio.csv",
        "data/limpieza_G3/ocq_limpio.csv",
        "data/limpieza_G3/slq_limpio.csv",
        "data/limpieza_G3/smq_limpio.csv"
    ]
    
    # Filter only existing files
    existing_files = [f for f in files if os.path.exists(f)]
    print(f"Archivos encontrados para unir: {len(existing_files)}")
    
    # Load and merge
    merged_df = None
    
    for file_path in existing_files:
        df = pd.read_csv(file_path)
        
        # In demo.csv, the column is named 'ID', rename to 'SEQN'
        if 'ID' in df.columns and 'SEQN' not in df.columns:
            df = df.rename(columns={'ID': 'SEQN'})
        
        if merged_df is None:
            merged_df = df
        else:
            # Check for duplicate columns (excluding SEQN)
            cols_to_use = df.columns.difference(merged_df.columns).tolist()
            if 'SEQN' not in cols_to_use:
                cols_to_use.append('SEQN')
                
            merged_df = pd.merge(merged_df, df[cols_to_use], on='SEQN', how='outer')
            
    # Remove rows with NaN in SEQN (if any)
    merged_df = merged_df.dropna(subset=['SEQN'])
    
    # Sort by SEQN
    merged_df = merged_df.sort_values('SEQN')
    
    print(f"Dataset final unido: {merged_df.shape}")
    
    # Save the result
    output_path = "data/dataset_completo.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"Dataset guardado en {output_path}")

if __name__ == "__main__":
    merge_cleaned_datasets()
