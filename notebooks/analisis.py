# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("../data/dataset_completo.csv")

print(df.columns)
# ['SEQN', 'Genero', 'Edad', 'Etnia', 'Nivel_Educativo', 'Estado_Civil', 'Ratio_Pobreza', 'Servicio_Militar', 'Ratio_Pobreza_Faltante', 'Consumio_Alcohol', 'Dias_4_5_Tragos', 'Dias_Con_+4_5_Tragos_En_2hrs', 'Dias_Consumo_Anual', 'Frec_Consumo_Anual', 'Promedio_Tragos_Dia', 'Unidad_4_5_Tragos', 'Unidad_Frec_Anual', 'enfermedad_coronaria', 'problema_higado', 'problema_tiroides', 'tiene_artritis', 'tiene_asma', 'tiene_psoriasis', 'tiene_sobrepeso', 'tuvo_ataque_corazon', 'tuvo_cancer', 'tuvo_derrame_cerebral', 'ejercicio_moderado_recreativo', 'ejercicio_vigoroso_recreativo', 'minutos_sedentario_dia', 'trabajo_moderado', 'trabajo_vigoroso', 'transporte_activo_bici_caminar', 'altura_pulgadas', 'conducta_fumar_para_adelgazar', 'conducta_laxantes_vomito', 'conducta_saltar_comidas', 'intento_bajar_peso', 'intento_bajar_peso_1año', 'percibe_peso_como', 'perdio_peso_intencional', 'peso_libras', 'BMDAVSAD', 'BMDSTATS', 'BMXARMC', 'BMXBMI', 'BMXHT', 'BMXLEG', 'BMXWAIST', 'BMXWT', 'CBQ505', 'CBQ535', 'CBQ540', 'CBQ545', 'DBD895', 'DBD900', 'DBD905', 'DBD910', 'DBQ197', 'DBQ229', 'DBQ700', 'DR1DAY', 'DR1DRSTZ', 'DR1TALCO', 'DR1TCAFF', 'DR1TCARB', 'DR1TFDFE', 'DR1TFIBE', 'DR1TIRON', 'DR1TKCAL', 'DR1TMAGN', 'DR1TNUMF', 'DR1TPROT', 'DR1TSFAT', 'DR1TSODI', 'DR1TSUGR', 'DR1TTFAT', 'DR1TVB12', 'DR1TVB6', 'DR1TVC', 'DR1TVD', 'DR1TZINC', 'DR1_320Z', 'DRDINT', 'DRQSDIET', 'DRQSDT1', 'DRQSDT10', 'DRQSDT3', 'DRQSDT4', 'DRQSDT7', 'DRQSDT9', 'WTDRD1', 'DUQ200', 'DUQ210', 'DUQ217', 'DUQ220Q', 'DUQ220U', 'DUQ230', 'DUQ240', 'DUQ250', 'DUQ260', 'DUQ280', 'DUQ290', 'DUQ300', 'DUQ320', 'DUQ330', 'DUQ340', 'DUQ360', 'DUQ370', 'DUQ390', 'DUQ410', 'DUQ420', 'DUQ430', 'DLQ010', 'DLQ020', 'DLQ040', 'DLQ050', 'DLQ060', 'DLQ080', 'OCD231', 'OCD241', 'OCD390G', 'OCD391', 'OCD392', 'OCD395', 'OCQ180', 'OCQ210', 'OCQ260', 'SLD010H', 'SLQ050', 'SLQ060', 'SMD030', 'SMD630', 'SMD641', 'SMD650', 'SMQ020', 'SMQ040', 'SMQ078', 'SMQ621', 'SMQ670', 'SMQ848', 'SMQ852Q', 'SMQ852U']
out = []
for col in df.columns:
    out.append(col)

df_alq = pd.read_csv('../data/alq.csv')
# %%
