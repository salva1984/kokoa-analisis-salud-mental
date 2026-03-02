# %%
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt

merged_df = pd.read_csv("data/dataset_completo.csv")

# %%
# Esto te genera una matriz donde las rayas blancas son los datos nulos
msno.matrix(merged_df)
plt.show()