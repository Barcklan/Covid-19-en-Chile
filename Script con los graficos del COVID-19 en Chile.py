#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 23:54:05 2024
SCRIPT CON LAS CODIGOS DE CADA GRAFICO QUE APARECE EN EL INFORME DE ANALISIS ESTADISTICO DE LAS DEFUNCIONES POR COVID-19
EN CHILE DESDE 2020 HASTA 2024.

@author: barcklan
"""
############################################################################################################
###### CARGAR LIBRERÍAS, DATOS Y GRAFICAR LA TEDENCIA DE DEFUNCIONES POR COVID-19 ############
############################################################################################################

# Importar librerías necesarias
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Carga el archivo excel (.xlsx)

data = pd.read_excel('/Users/barcklan/Desktop/Escritorio Python - A/defunciones_covid19_2020_2024.xlsx')

#Eliminar registros con edad superior a los 103 años

data_filtrada = data[data['EDAD_CANT']<=103]


# Especifico las columnas que deseo conservar en la dataframe

columnas_deseadas = ['AÑO','FECHA_DEF', 'SEXO_NOMBRE', 'EDAD_CANT', 'COMUNA', 'NOMBRE_REGION', 'DIAG1',
                                  'GLOSA_SUBCATEGORIA_DIAG1', 'LUGAR_DEFUNCION']

# Creo un dataframe sólo con las columnas consideradas en la variable "columnas_deseadas"

data_filtrada = data_filtrada[columnas_deseadas]


#Agrupar por AÑO, NOMBRE_REGION y COMUNA y contar los fallecimientos

comunas_muertes = data_filtrada.groupby(['AÑO', 'NOMBRE_REGION', 'COMUNA', 'SEXO_NOMBRE', 'DIAG1', 'LUGAR_DEFUNCION']).size().reset_index(name='MUERTES')


#Asegurarse de que la columna FECHA_DEF esté en formato de fecha
data_filtrada['FECHA_DEF'] = pd.to_datetime(data_filtrada['FECHA_DEF'], errors='coerce')

#Extraer el año y el mes para análisis mensual
data_filtrada['AÑO_MES'] = data_filtrada['FECHA_DEF'].dt.to_period('M')

# Conteo de fallecidos por mes
muertes_mensuales = data_filtrada.groupby('AÑO_MES').size()

# Gráfico de tendencias de fallecimientos
sns.set(style="darkgrid")
muertes_mensuales.plot(kind='line', figsize=(10,5))
plt.title('Tendencia de Defunciones por COVID-19', fontsize=14)
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Número de Defunciones', fontsize=12)
plt.savefig('Tendencia_muertes_en_el_tiempo.png', dpi=300, bbox_inches='tight')  # Guardar opcional
# Guardar el gráfico en alta calidad como PNG
plt.show()

############################################################################################################
####################################  FIN DEL COMANDO ##############################################################
############################################################################################################

########################################################################################################################
############## GRAFICO MUERTES % ACUMULADOS POR COVID-19 A NIVEL DE REGIONES  ##############################
########################################################################################################################

comunas_muertes = data_filtrada.groupby(['AÑO', 'NOMBRE_REGION', 'COMUNA', 'SEXO_NOMBRE', 'DIAG1', 'LUGAR_DEFUNCION']).size().reset_index(name='MUERTES')


# Asegúrate de tener la columna 'NOMBRE_REGION' en tu DataFrame
df1 = comunas_muertes.groupby('NOMBRE_REGION').agg({'MUERTES': 'sum'}).reset_index()


# Calcular el porcentaje acumulado de muertes por región
df1['% Acumulado'] = (df1['MUERTES'] / df1['MUERTES'].sum()) * 100

# Ordenar de mayor a menor según el número de muertes
df1 = df1.sort_values('MUERTES', ascending=False)

# Crear gráfico
sns.set(style="whitegrid")
plt.figure(figsize=(10, 8))

ax = sns.barplot(
    data=df1, 
    y='NOMBRE_REGION', 
    x='MUERTES', 
    palette='Blues_d'
)

# Añadir porcentaje y número de muertes a las barras horizontales
for p in ax.patches:
    width = p.get_width()
    percentage = df1.loc[df1['MUERTES'] == width, '% Acumulado'].values[0]
    # Colocar las etiquetas en la parte superior de las barras
    ax.annotate(f'{width:.0f} ({percentage:.1f}%)',
                (width + 500, p.get_y() + p.get_height() / 2),  # Ajuste de posición para etiquetas
                ha='left', va='center', color='black', fontsize=10)

# Personalización
plt.title("Muertes % acumulados por Covid-19 a nivel de regiones", fontsize=16)
plt.xlabel("Número de muertes", fontsize=12)
plt.ylabel("Región", fontsize=12)
plt.savefig('Muertes_acumulados_regiones.png', dpi=300, bbox_inches='tight')
plt.show()

########################################################################################################################
############################## FIN DEL COMANDO ############################################################
########################################################################################################################



################################################################################################################
################ CREACIÓN DE HISTOGRAMA DE LA VARIABLE EDAD DE LOS PACIENTES DESDE EL 2020 HASTA 2024 ##########
################################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

comunas_muertes = data_filtrada.groupby(['AÑO', 'NOMBRE_REGION', 'COMUNA', 'SEXO_NOMBRE', 'DIAG1', 'LUGAR_DEFUNCION']).size().reset_index(name='MUERTES')

#######################################################
######### HISTOGRAMA AÑO 2020 COMO REFERENCIA  ########
#######################################################

# Filtrar los datos solo para el año 202X si es necesario (ir cambiando el año por el solicitado)
df_2020 = data_filtrada[data_filtrada['AÑO'] == 2020]  # Asegúrate de tener la columna 'AÑO'

# Configuración de estilo
sns.set(style="white")

# Crear histograma
plt.figure(figsize=(10, 6))  # Tamaño de la figura
sns.histplot(
    data=df_2020, 
    x='EDAD_CANT', 
    bins=15,           # Número de bins en el histograma
    kde=True,          # Añadir densidad suavizada
    color='skyblue',   # Color de las barras
    edgecolor='black',
    linewidth=1.5,
    line_kws={'color' : 'red', 'linewidth': 2}  # Grosor de la línea KDE
)

# Personalizar el gráfico
plt.title("Distribución de Edad de los Fallecidos durante el año 2020", fontsize=14) # Cambiar el año por el solicitado
plt.xlabel("Edad", fontsize=12)
plt.ylabel("Frecuencia", fontsize=12)

plt.savefig('Distribución_Edad_2020.png', dpi=300, bbox_inches='tight')  # Cambiando el el nombre o el año por el solicitado

# Mostrar gráfico
plt.show()


#####################################################################################################################
######### GRÁFICO CON EL PROMEDIO DE MUERTES COVID-19 POR AÑO Y REGIÓN (ESCALA LOGARITMICA)  ###############################################
#####################################################################################################################


# Agrupar por 'AÑO' y 'NOMBRE_REGION', sumando el total de muertes
df_total_muertes = comunas_muertes.groupby(['AÑO', 'NOMBRE_REGION'], as_index=False)['MUERTES'].sum()

# Calcular el promedio global anual
promedio_global_anual = df_total_muertes.groupby('AÑO')['MUERTES'].mean().reset_index()
promedio_global_anual['NOMBRE_REGION'] = 'Promedio Global'

# Calcular el Gran Total promedio de muertes
gran_total = df_total_muertes['MUERTES'].mean()

# Crear el DataFrame completo con el promedio global
df_grafico = pd.concat([df_total_muertes, promedio_global_anual], ignore_index=True)

# Paleta de colores personalizada para cada región
colores_regiones = {
    'Metropolitana de Santiago': 'gray',
    'De Valparaíso': 'brown',
    'Del Bíobío': 'pink',
    'Del Maule': 'purple',
    'De La Araucanía': 'green',
    'De Tarapacá': 'orange',
    'De Los Lagos': 'red',
    'De Los Ríos': 'darkblue',
    'De Ñuble': 'magenta',
    'De Coquimbo': 'lightgreen',
    'De Magallanes y de La Antártica Chilena': 'blue',
    'De Antofagasta': 'darkorange',
    'De Arica y Parinacota': 'teal',
    'De Atacama': 'coral',
    'De Aisén del Gral. C. Ibáñez del Campo': 'navy',
    'Del Libertador B. O\'Higgins': 'violet',
    'Promedio Global': 'black'  # Color del promedio global
}

# Configuración del estilo
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

# Graficar cada región con la paleta de colores personalizada
for region in df_grafico['NOMBRE_REGION'].unique():
    data_region = df_grafico[df_grafico['NOMBRE_REGION'] == region]
    color = colores_regiones.get(region, 'gray')  # Asignar color o gris por defecto
    linestyle = '-' if region == 'Promedio Global' else '--'  # Línea continua para promedio global
    linewidth = 2 if region == 'Promedio Global' else 1
    plt.plot(data_region['AÑO'], data_region['MUERTES'],
             label=region, color=color, linestyle=linestyle, linewidth=linewidth, marker='o')

# Agregar la línea horizontal del Gran Total
plt.axhline(y=gran_total, color='black', linestyle='--', linewidth=2, 
            label=f'Gran Total: {gran_total:.2f}')

# Eliminar duplicados de la leyenda
handles, labels = plt.gca().get_legend_handles_labels()
unique_legend = dict(zip(labels, handles))
plt.legend(unique_legend.values(), unique_legend.keys(),
           title="Región / Promedio Global", bbox_to_anchor=(1.05, 1), loc='upper left')

# Títulos y etiquetas
plt.title("Promedio de muertes por COVID-19 por año y región (Escala logarítmica)", fontsize=16)
plt.xlabel("Año", fontsize=12)
plt.ylabel("Promedio de muertes (escala logarítmica)", fontsize=12)

# Escala logarítmica en el eje Y
plt.yscale("log")
plt.savefig('Promedio_muertes_x_año.png', dpi=300, bbox_inches='tight')

# Mostrar el gráfico
plt.tight_layout()
plt.show()

################################################################################################
##################  FIN DE COMANDO  ############################################################
################################################################################################

############################################################################################################################################################
############### GRAFICO DE BARRAS AGRUPADAS CON EL TOP 3 COMUNAS CON MAYOR NUMERO DE FALLECIDOS POR AÑO #############
############################################################################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Crear un DataFrame
df = comunas_muertes

# 1. Filtrar las columnas necesarias
df_filtrado = df[['AÑO', 'COMUNA', 'MUERTES']]

# 2. Agrupar por 'AÑO' y 'COMUNA', sumando el total de muertes
df_agrupado = df_filtrado.groupby(['AÑO', 'COMUNA'], as_index=False).sum()

# 3. Seleccionar el top 3 de comunas con más muertes por año
top3_comunas = df_agrupado.groupby('AÑO', group_keys=False).apply(lambda x: x.nlargest(3, 'MUERTES'))

# Mostrar el DataFrame resultante
print(top3_comunas)



# Configuración del estilo
sns.set(style="whitegrid")
plt.figure(figsize=(14, 8))  # Aumentar tamaño

# Crear el gráfico
ax = sns.barplot(
    x='AÑO', 
    y='MUERTES', 
    hue='COMUNA', 
    data=top3_comunas,
    palette='tab10'
)

# Centrar las etiquetas del eje X
# Se obtienen las posiciones actuales de las etiquetas y se ajustan
new_xticks = sorted(top3_comunas['AÑO'].unique())
ax.set_xticks(range(len(new_xticks)))  # Define nuevas posiciones centradas
ax.set_xticklabels(new_xticks)  # Reasigna las etiquetas


# Agregar etiquetas encima de todas las barras
for p in ax.patches:
    height = p.get_height()
    if height > 0:  # Evitar etiquetas en barras vacías
        ax.annotate(f'{int(height)}', 
                    (p.get_x() + p.get_width() / 2, height + 5),  # Ajuste vertical
                    ha='center', va='bottom', fontsize=10, color='black')

# Configuración de títulos y ejes
plt.title("Top 3 comunas con mayor número de fallecidos por COVID-19 por año", fontsize=16, weight='bold')
plt.xlabel("Año", fontsize=12)
plt.ylabel("Total de fallecidos", fontsize=12)

# Leyenda
plt.legend(title="Comuna", loc='upper right')

plt.savefig('top3_comunas.png', dpi=300, bbox_inches='tight')

# Mostrar el gráfico
plt.tight_layout()
plt.show()

############################################################################################################################################################
################################## FIN DEL CODIGO ################################################################
############################################################################################################################################################

#####################################################################################################################
###################### GRAFICO DE MUERTES POR COVID-19 SEGUN LUGAR DE DEFUNCION (2020-2024)  ##########################
#####################################################################################################################

import pandas as pd

# Suponiendo que 'df' es tu DataFrame original
df_agrupado = df.groupby(['AÑO', 'LUGAR_DEFUNCION'], as_index=False)['MUERTES'].sum()

print(df_agrupado.isnull().sum())


df_pivot = df_agrupado.pivot(index='AÑO', columns='LUGAR_DEFUNCION', values='MUERTES').fillna(0)

import seaborn as sns
import matplotlib.pyplot as plt

# Configuración del estilo
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(12, 6))

# Paleta de colores personalizada
colores = {
    'Hospital o Clínica': 'orangered',
    'Casa habitación': 'royalblue',
    'Otro': 'green'
}

# Graficar cada lugar de defunción
for lugar, color in colores.items():
    if lugar in df_pivot.columns:
        plt.plot(df_pivot.index, df_pivot[lugar], 
                 label=lugar, color=color, linestyle='--', marker='o')

        # Añadir etiquetas de valores sobre cada punto
        for x, y in zip(df_pivot.index, df_pivot[lugar]):
            plt.text(x, y, f'{int(y)}', color=color, ha='center', va='bottom', fontsize=10)

# Configuración de la escala logarítmica en el eje Y
plt.yscale('log')

# Configurar títulos y etiquetas
plt.title("Distribución de muertes por COVID-19 según lugar de defunción (2020-2024)", fontsize=16, weight='bold')
plt.xlabel("Año", fontsize=12)
plt.ylabel("Número de muertes (escala logarítmica)", fontsize=12)

# Colocar la leyenda fuera del gráfico
plt.legend(title="Lugar de defunción", bbox_to_anchor=(1.05, 1), loc='upper left')

# Ajustar los márgenes para acomodar la leyenda
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Deja espacio para la leyenda a la derecha

# Mostrar gráfico
plt.show()

###########################################################################################
########################## FIN DEL CODIGO ####################################################
###########################################################################################

######################################################################################################################################################################################
########### GRAFICO DE LINEA CON LA DISTRIBUCION DE MUERTES POR COVID-19 SEGUN TIPO DE DEFUNCION  #############
######################################################################################################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Crear un DataFrame de ejemplo
data = {
    'AÑO': [2020, 2021, 2022, 2023, 2024],
    'U071': [16176.0, 21628.0, 12353.0, 2451.0, 710.0],
    'U072': [2484.0, 1229.0, 701.0, 119.0, 78.0],
    'U099': [0.0, 75.0, 87.0, 42.0, 29.0]
}
df = pd.DataFrame(data)

# Configurar 'AÑO' como índice
df.set_index('AÑO', inplace=True)

# Reemplazar ceros con NaN para evitar la línea segmentada
df = df.replace(0, np.nan)

# Configuración de estilo
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(12, 6))

# Paleta de colores personalizada
colores = {
    'U072': 'darkorange',
    'U071': 'darkblue',
    'U099': 'green'
}

# Graficar columnas con valores válidos
for col in df.columns:
    plt.plot(df.index, df[col], 
             label=col, color=colores.get(col, 'gray'), linestyle='--', marker='o')
    
    # Añadir etiquetas solo a puntos válidos
    for x, y in zip(df.index, df[col]):
        if not np.isnan(y):  # Mostrar solo valores válidos
            plt.text(x, y, f'{int(y)}', color=colores.get(col, 'gray'), 
                     ha='center', va='bottom', fontsize=10)

# Configurar escala logarítmica en el eje Y
plt.yscale('log')

# Configuración de título y etiquetas
plt.title("Distribución de muertes por COVID-19 según tipo de defunción", fontsize=16, weight='bold')
plt.xlabel("Año", fontsize=12)
plt.ylabel("Número de muertes (escala logarítmica)", fontsize=12)

# Colocar la leyenda fuera del gráfico
plt.legend(title="Tipo de defunción", bbox_to_anchor=(1.05, 1), loc='upper left')

# Ajustar diseño
plt.tight_layout()
# guardar el grafico en formato png con una calidad de 300 dpi
plt.savefig('distribucion_tipo_defuncion.png', dpi=300, bbox_inches='tight')

plt.show()


######################################################################################################################################################################################
#################################  FIN DEL CODIGO  ##########################################################
######################################################################################################################################################################################

#####################################################################################################################################################################
########################### GRAFICO DE BARRAS AGRUPADAS CON LAS MUERTES POR COVID-19 SEGUN GRUPO ETARIO Y SEXO ##################################################################
#####################################################################################################################################################################


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Carga el archivo excel (.xlsx)

data = pd.read_excel('/Users/barcklan/Desktop/Escritorio Python - A/defunciones_covid19_2020_2024.xlsx')

#Eliminar registros con edad superior a los 103 años

data_filtrada = data[data['EDAD_CANT']<=103]


# Especifico las columnas que deseo conservar en la dataframe

columnas_deseadas = ['AÑO','FECHA_DEF', 'SEXO_NOMBRE', 'EDAD_CANT', 'COMUNA', 'NOMBRE_REGION', 'DIAG1',
                                  'GLOSA_SUBCATEGORIA_DIAG1', 'LUGAR_DEFUNCION']

# Creo un dataframe sólo con las columnas consideradas en la variable "columnas_deseadas"

data_filtrada = data_filtrada[columnas_deseadas]

# Clasificar edades en grupos etarios (FUNCIÓN)
def clasificar_grupo_etario(edad):
    if edad <= 17:
        return '1. Niños y Adolescentes'
    elif 18 <= edad <= 29:
        return '2. Jóvenes'
    elif 30 <= edad <= 44:
        return '3. Adultos Jóvenes'
    elif 45 <= edad <= 59:
        return '4. Adultos de Mediana Edad'
    elif 60 <= edad <= 79:
        return '5. Tercera Edad'
    else:
        return '6. Cuarta Edad'



#### Creamos una columna en la dataframe "data_filtrada" llamada "Grupo Etario" aplicando 
#la función de clasificación de edades en grupos etarios

data_filtrada['Grupo Etario'] = data_filtrada['EDAD_CANT'].apply(clasificar_grupo_etario)


#Imprimimos la tabla data_filtrada

print(data_filtrada)


# Creo una variable llamada comunas_muertes_B para agrupar las variables del penultimo comando pero agregando la variable grupo etario

comunas_muertes_B = data_filtrada.groupby(['AÑO', 'NOMBRE_REGION', 'COMUNA', 'SEXO_NOMBRE', 'DIAG1', 'LUGAR_DEFUNCION', 'Grupo Etario']).size().reset_index(name='MUERTES')


# Paso 1: Agrupar los datos
# Agrupar por 'Grupo Etario' y 'SEXO_NOMBRE', sumando las muertes
df_grouped = comunas_muertes_B.groupby(['Grupo Etario', 'SEXO_NOMBRE'], as_index=False)['MUERTES'].sum()

# Paso 2: Reorganizar para compatibilidad con Seaborn
# Dejarlo en formato ordenado
df_grouped = df_grouped.sort_values(by='Grupo Etario')

# Paso 3: Configuración del gráfico
sns.set(style="whitegrid", font_scale=1.2)  # Estilo y tamaño de fuente
plt.figure(figsize=(12, 6))  # Tamaño del gráfico

# Crear el gráfico de barras agrupadas
ax = sns.barplot(data=df_grouped, x='Grupo Etario', y='MUERTES', hue='SEXO_NOMBRE', 
                 palette=['#4169E1', '#CD7F32'])  # Azul y café

# Paso 4: Añadir etiquetas a las barras
for p in ax.patches:
    height = int(p.get_height())
    if height > 0:  # Evita etiquetas para barras con altura cero
        ax.annotate(f'{height}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', fontsize=10, color='black')

# Paso 5: Configuración adicional
plt.title("Distribución de Muertes por COVID-19 según Grupo Etario y Sexo", fontsize=16, weight='bold')
plt.xlabel("Grupo Etario", fontsize=12)
plt.ylabel("Total de Muertes", fontsize=12)

plt.xticks(rotation=45, ha='right')  # Rotar etiquetas del eje X
plt.legend(title="Sexo", bbox_to_anchor=(1.05, 1), loc='upper left')  # Mover leyenda fuera del gráfico


plt.tight_layout()  # Ajustar los márgenes

plt.savefig('distribucion_grupo_etario_sexo.png', dpi=300, bbox_inches='tight')

plt.show()


####################################################################################################################################
#################################  FIN DEL CODIGO ##################################################################
####################################################################################################################################


##################################################################################################################################################################### 
###########################  GRAFICO DE BARRAS CON EL PORCENTAJE DE MUERTES POR COVID-19 ENTRE HOMBRES Y MUJERES ############################################## 
#####################################################################################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Crear el DataFrame con tus datos (reemplázalo con tu DataFrame completo)
data = {
    'SEXO_NOMBRE': ['Hombre', 'Mujer'],
    'MUERTES': [55.7, 44.3]  # Porcentajes ya calculados
}

df = pd.DataFrame(data)

# Configuración de Seaborn y Matplotlib
sns.set(style="whitegrid", font_scale=1.2)  # Estilo y tamaño de fuente
plt.figure(figsize=(8, 6))  # Tamaño del gráfico

# Crear el gráfico de barras con Seaborn
ax = sns.barplot(data=df, x='SEXO_NOMBRE', y='MUERTES', 
                 palette=['#A9CCE3', '#F5CBA7'],  # Colores pastel
                 edgecolor='black')

# Añadir etiquetas a las barras
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=12, color='black')

# Configurar título y etiquetas
plt.title("Porcentaje de muertes por COVID-19 entre hombres y mujeres (2020-2024)", 
          fontsize=16, weight='bold')
plt.xlabel("Género", fontsize=14)
plt.ylabel("Porcentaje de muertes (%)", fontsize=14)

# Ajustar el diseño
plt.tight_layout()
plt.savefig('porcentaje de muertes_por_sexo.png', dpi=300, bbox_inches='tight')
plt.show()


###################################################################################################
################################# FIN DEL CODIGO ##################################################
###################################################################################################

#####################################################################################################################################################################
############################# GRAFICO BOX-PLOT CON LA DISTRIBUCIÓN DE PACIENTES FALLECIDOS POR COVID-19 (2020 - 2024) #################################
#####################################################################################################################################################################

# Carga el archivo excel (.xlsx)

data0 = pd.read_excel('/Users/barcklan/Desktop/Escritorio Python - A/defunciones_covid19_2020_2024.xlsx')


df = data0


# Crear el grafico
plt.figure(figsize=(12,6))
sns.boxplot(x='AÑO', y='EDAD_CANT', data=df, palette='coolwarm')

# Añadir la línea de edad límite
plt.axhline(y=103, color='red', linestyle='--', linewidth=2, label='Edad límite (103 años)')



# Añadir título y etiquetas
plt.title('Distribución de Edades de Pacientes Fallecidos por COVID-19 (2020-2024)', fontsize=16)
plt.xlabel('AÑO', fontsize=14)
plt.ylabel('Edad', fontsize=14)

# Añadir la leyenda
plt.legend(loc='upper left', bbox_to_anchor=(1,1))

# Guardar la imagen en forma .png
plt.savefig('boxplot_edad.png', dpi=300, bbox_inches='tight')

# Mostrar el gráfico
plt.show()

#######################################################################################
############################# FIN DEL CODIGO ########################################## 
#######################################################################################


###################################################################################################################################################
######################### GRAFICO DE LINEAS CON LA EVOLUCION DE MUERTES POR COVID-19 ENTRE HOMBRES Y MUJERES (2020 - 2024) #####################
###################################################################################################################################################

df = comunas_muertes_B

# Agrupar los datos por AÑO y SEXO_NOMBRE y sumar las muertes
df_grouped = df.groupby(['AÑO', 'SEXO_NOMBRE'])['MUERTES'].sum().reset_index()

# Configuración de estilo
sns.set(style='whitegrid')

# Crear el gráfico
plt.figure(figsize=(10,6))
sns.lineplot(x='AÑO', y='MUERTES', hue='SEXO_NOMBRE', data=df_grouped, marker='o', palette=['skyblue', 'lightcoral'])

# Añadir ey¡tiquetas a cada punto
for line in df_grouped['SEXO_NOMBRE'].unique():
    subset = df_grouped[df_grouped['SEXO_NOMBRE'] == line]
    for x,y in zip(subset['AÑO'], subset['MUERTES']):
        plt.text(x, y, f'{y}', horizontalalignment='center', verticalalignment='bottom' if line == 'Hombre' else 'top',
                 color='blue' if line == 'Hombre' else 'red',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
plt.xlabel('Año')
plt.ylabel('Total de muertes')
plt.title('Evolución de muertes por COVID-19 entre hombres y mujeres (2020 - 2024)')

# Añadir la leyenda fuera del area del grafico
plt.legend(title='SEXO_NOMBRE', bbox_to_anchor=(1.05, 1), loc='upper left')


# Guardar el gráfico en alta calidad
plt.tight_layout()
plt.savefig('evolucion_muertes_genero_covid_etiquetas.png', dpi=300, bbox_inches='tight')

# Mostrar el gráfico
plt.show()

#########################################################################################################
##################### FIN DEL CODIGO ###############################################################
#########################################################################################################


###################################################################################################################################################
##################### GRAFICO DE LINEAS CON LA EVOLUCION DE MUERTES POR COVID-19 POR REGION DESDE (2020 - 2024) #####################
###################################################################################################################################################


df=data0

# Agrupar los datos por AÑO y NOMBRE_REGION y sumar las muertes
df_grouped = df.groupby(['AÑO', 'NOMBRE_REGION'])['MUERTES'].sum().reset_index()

# Crear el grafico
plt.figure(figsize=(12, 6))

# Iterar sobre cada región para crear las líneas del gráfico
for region in df_grouped['NOMBRE_REGION'].unique():
    subset = df_grouped[df_grouped['NOMBRE_REGION'] == region]
    plt.plot(subset['AÑO'], subset['MUERTES'], marker='o', label=region)
    
    
    # Añadir título y etiquetas
plt.title('Evolución de muertes por COVID-19 por región desde 2020 - 2024')
plt.xlabel('AÑO')
plt.ylabel('Número de muertes')

# Añadir leyenda fuera del área del gráfico
plt.legend(title='Región', bbox_to_anchor=(1.05, 1), loc='upper left')

   # Mostrar el gráfico
plt.tight_layout()
plt.savefig('evolucion_muertes_regiones_covid_etiquetas.png', dpi=300, bbox_inches='tight')
plt.show()
    
#########################################################################################################
##################### FIN DEL CODIGO ###############################################################
#########################################################################################################    
    




