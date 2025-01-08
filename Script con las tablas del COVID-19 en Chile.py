# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:53:28 2024
SCRIPT CON LA CREACION DE CADA TABLA ADJUNTADA EN EL INFORME DE ANALISIS ESTADISTICO DE LAS DEFUNCIONES POR COVID-19
EN CHILE DESDE 2020 HASTA 2024.

@author: barcklan
"""
############################################################################
############ CARGAR LAS LIBRERIAS NECESARIAS PARA LOS ANALISIS  #############
#############################################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

################################################################################################################################################
##################################### FIN DE COMAMDOS LIBRERIAS ########################################################################
################################################################################################################################################

################################################################################################################################################
############## CARGAR LOS DATOS APARTIR DE UN ARCHIVO EXCEL (.XLSX) ########################################################################
################################################################################################################################################

# Carga el archivo excel (.xlsx)

data = pd.read_excel('/Users/barcklan/Desktop/Escritorio Python - A/defunciones_covid19_2020_2024.xlsx')

################################################################################################################################################
############################# ELIMINAR LOS REGISTROS CUYA EDAD DE LOS PACIENTES SEA MAYOR O IGUAL A 103 AÑOS  ###########################################
################################################################################################################################################


#Eliminar registros con edad superior a los 103 años

data_filtrada = data[data['EDAD_CANT']<=103]

################################################################################################################################################
################################ FIN DEL COMANDO  ########################################
################################################################################################################################################


################################################################################################################################################
################################## CONSERVAR LAS COLUMNAS SELECCIONADAS EN EL DATAFRAME  ######################################
################################################################################################################################################

# Especifico las columnas que deseo conservar en la dataframe

columnas_deseadas = ['AÑO','FECHA_DEF', 'SEXO_NOMBRE', 'EDAD_CANT', 'COMUNA', 'NOMBRE_REGION', 'DIAG1',
                                  'GLOSA_SUBCATEGORIA_DIAG1', 'LUGAR_DEFUNCION']

# Creo un dataframe sólo con las columnas consideradas en la variable "columnas_deseadas"

data_filtrada = data_filtrada[columnas_deseadas]

################################################################################################################################################
###################################### FIN DEL COMANDO SELECCIONAR COLUMNAS SELECCIONADAS  ##################################
################################################################################################################################################

# Conteo de fallecidos por sexo

print(data_filtrada['SEXO_NOMBRE'].value_counts())

########################################################################################################################################################################################################################
#################################### AGRUPAR LOS DATOS POR  LAS COLUMNAS SELCCIONADAS Y CREAR Y SUMAR A SU VEZ UNA NUEVA COLUMNA LLAMADA MUERTES   ####################################
########################################################################################################################################################################################################################

#Agrupar por AÑO, NOMBRE_REGION y COMUNA y contar los fallecimientos

comunas_muertes = data_filtrada.groupby(['AÑO', 'NOMBRE_REGION', 'COMUNA', 'SEXO_NOMBRE', 'DIAG1', 'LUGAR_DEFUNCION']).size().reset_index(name='MUERTES')


# Ver las estadisticas descriptivas de la variable Muertes (media, desv. estandar, etc...)

comunas_muertes['MUERTES'].describe()


 ################################################################################################################################################
############################ FIN DEL COMANDO AGRUPAR LOS DATOS .....   ############################################
################################################################################################################################################


################################################################################################################################################
############################### CREACION DE LA FUNCION QUE ME PERMITA CLASIFICAR LA EDAD DE LOS PACIENTES POR GRUPO ETARIO  #########################################
################################################################################################################################################

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

################################

### Aqui lo mismo agregamos una columna llamada "Grupo Etario" pero en la dataframe llamada comunas_muertes_B 

#comunas_muertes_B['Grupo Etario'] = comunas_muertes['EDAD_CANT'].apply(clasificar_grupo_etario)

#print(comunas_muertes_B)

################################################################################################################################################
################## FIN DEL COMANDO DE LA CREACIÓN DE UNA FUNCIÓN QUE ME PERMITA CLASIFICAR LA EDAD DE LOS PACIENTES POR GRUPO ETARIO ##################
################################################################################################################################################



################################################################################################################################################
####### OBTENCIÓN DE LA TABLA DE FRECUENCIA CON LAS PROPORCIONES DE HOMBRES Y MUJERES FALLECIDOS POR COVID-19 #######################################################################
################################################################################################################################################

### Agrupamos por región y año, luego sumamos las muertes
df_agrupado = comunas_muertes.groupby(['NOMBRE_REGION', 'AÑO']).agg({'MUERTES': 'sum'}).reset_index()

# Agrupamos por región y año, luego sumamos las muertes
df_agrupado_A = comunas_muertes.groupby(['NOMBRE_REGION', 'AÑO', 'SEXO_NOMBRE', 'DIAG1', 'LUGAR_DEFUNCION', 'COMUNA']).agg({'MUERTES': 'sum'}).reset_index()

# Paso 1: Calcular la frecuencia absoluta
frecuencia_absoluta = data_filtrada['SEXO_NOMBRE'].value_counts()

# Paso 2: Calcular la frecuencia relativa
frecuencia_relativa = data_filtrada['SEXO_NOMBRE'].value_counts(normalize=True)

# Paso 3: Calcular la frecuencia absoluta acumulada
frecuencia_acumulada = frecuencia_absoluta.cumsum()

# Paso 4: Calcular la frecuencia porcentual
frecuencia_porcentual = frecuencia_relativa * 100

# Paso 5: Crear un DataFrame con los resultados
tabla_frecuencias = pd.DataFrame({
    'Frecuencia Absoluta': frecuencia_absoluta,
    'Frecuencia Relativa': frecuencia_relativa,
    'Frecuencia Acumulada': frecuencia_acumulada,
    'Frecuencia Porcentual': frecuencia_porcentual
})

# Mostrar la tabla de frecuencias
print(tabla_frecuencias)

########################################################################
############################# FIN COMANDO TABLA ######################
########################################################################



############################################################################################
############# CREACIÓN DE LA TABLA CON LAS MUERTES POR COVID-19 SEGÚN GRUPO ETARIOS ########

# Sumar las muertes por grupo etario y sexo
df_suma = comunas_muertes_B.groupby(['Grupo Etario', 'SEXO_NOMBRE'])['MUERTES'].sum().reset_index()

print(df_suma)

# Calcular el total global
total_global = df_suma['MUERTES'].sum()

# Agregar el porcentaje respecto al total global
df_suma['Porcentaje Global (%)'] = (df_suma['MUERTES'] / total_global) * 100

##############################################################
############################ FIN COMANDO TABLA ###############
##############################################################

###############################################################################################
##################### CREAR LA TABLA PARA EL TIPO DE DEFUNCIÓN U071; U072; U099  ###############
################################################################################################

# Crear la tabla con la función pivot_table
tabla = comunas_muertes_B.pivot_table(index="AÑO", columns="DIAG1", values="MUERTES", aggfunc="sum", fill_value=0)

# Renombrar las columnas (opcional)
tabla.columns.name = None  # Eliminar nombre de la columna
tabla.reset_index(inplace=True)  # Hacer que el índice (Año) sea una columna

print(tabla)

##############################################################################################
#############################################  FIN DE LA TABLA ################################
###############################################################################################


###############################################################################
###############  CREAR TABLA PARA EL LUGAR DE DEFUNCIÓN DE PACIENTES ###########################
###############################################################################

# Crear la tabla con la función pivot_table
tabla2 = comunas_muertes_B.pivot_table(index="AÑO", columns="LUGAR_DEFUNCION", values="MUERTES", aggfunc="sum", fill_value=0)

# Renombrar las columnas (opcional)
tabla2.columns.name = None  # Eliminar nombre de la columna
tabla2.reset_index(inplace=True)  # Hacer que el índice (Año) sea una columna

print(tabla2)

#####################################################################
####################### FIN DE LA TABLA #######################################################
#####################################################################################


#####################################################################################################
################# CREAR TABLA CON EL PROMEDIO DE MUERTES CADA AÑO (2020 - 2024) POR REGIÓN ##########
#####################################################################################################

# Crear tabla pivote con regiones en las filas y años como columnas
tabla3 = comunas_muertes_B.pivot_table(index="NOMBRE_REGION", columns="AÑO", values="MUERTES", aggfunc="sum", fill_value=0)

# Calcular el promedio de muertes por región (eje horizontal - años)
tabla3["Promedio"] = tabla3.mean(axis=1)

# Eliminar el nombre de las columnas (opcional)
tabla3.columns.name = None

print(tabla3)

promedio_global = tabla3.mean(axis=0)

tabla3.loc["Promedio Global"] = promedio_global

################################################################################
########################## FIN DE LA TABLA #####################################
################################################################################

################################################################################################################################################
########### CREAR TABLA CON LOS TOP 3 COMUNAS CON MAYOR NUMERO DE FALLECIDOS POR COVID-19 CADA AÑO  #########
################################################################################################################################################
# Datos simulados: Número de fallecidos por comuna y año
data1 = {
    "Año": [2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2022, 2023, 2023, 2023, 2024, 2024, 2024],
    "Comuna": [
        "Puente Alto", "Maipú", "La Florida",
        "Puente Alto", "Maipú", "La Florida",
        "Puente Alto", "Maipú", "La Florida",
        "Puente Alto", "Maipú", "Valparaíso",
        "Maipú", "La Florida", "Chillán"
    ],
    "Fallecidos": [
        931, 840, 639, 771, 666, 539, 329, 281, 300, 90, 72, 61, 25, 23, 22
    ]
}

# Crear DataFrame
df = pd.DataFrame(data1)

# Crear tabla pivote con comunas en las filas y años como columnas
tabla4 = df.pivot_table(index="Comuna", columns="Año", values="Fallecidos", aggfunc="sum", fill_value=0)

# Ordenar las filas según el total de fallecidos
tabla4["Total Fallecidos"] = tabla.sum(axis=1)
tabla4 = tabla4.sort_values("Total Fallecidos", ascending=False)

# Eliminar la columna "Total Fallecidos" si no se quiere mostrar
# tabla = tabla.drop(columns=["Total Fallecidos"])

print(tabla4)
########################################################################
####################  FIN DE LA TABLA #########################################
#########################################################################


################################################################################################################################################
################## CREAR TABLA CON LAS MUERTES ACUMULADAS POR COVID-19 POR REGIÓN DESDE 2020 HASTA 2024 ################
################################################################################################################################################


# Crear el DataFrame
df1 = comunas_muertes_B.pivot_table(index="NOMBRE_REGION", values="MUERTES", aggfunc="sum", fill_value=0)


# Calcular el porcentaje acumulado
df1["% Acumulado"] = (df1["MUERTES"] / df1["MUERTES"].sum()) * 100

# Ordenar por número de muertes (descendente)
df1 = df1.sort_values("MUERTES", ascending=False)

# Formatear la columna del porcentaje con 1 decimal
df1["% Acumulado"] = df1["% Acumulado"].map(lambda x: f"{x:.1f}%")

# Mostrar la tabla
print(df1)

################################################################################################################################################
######################################################################## FIN DE LA TABLA ########################################################################
################################################################################################################################################


    
    








