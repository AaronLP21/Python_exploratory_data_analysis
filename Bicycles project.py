# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 20:01:26 2021

@author: Aaron
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
#from textblob import TextBlob
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import datetime
sns.set()
os.getcwd()
os.chdir('C:\\Users\\Aaron\\Desktop\\Boehringer\\Codes\\Python\\Miros')
os.getcwd()
data=pd.read_csv("dataset.csv")
#Inspeccion de lo sdatos
list(data.columns)
data.head()
data.shape
data.info()
data.describe()
#mayor detalle de las variables:
data['Subscriber Type'].value_counts()
data['Start Station'].value_counts()
data['End Station'].value_counts()

#pivot tables
pivot1 = pd.pivot_table(data.iloc[:,[3,9]], index = 'Start Station', aggfunc='count')
print(pivot1)#total de suscriptores por terminal

pivot2 = pd.pivot_table(data, index = ['Start Station','Subscriber Type'],
                        aggfunc='count', margins=True)
print(pivot2)#debemos recortar esta tabla
pivot2.head()
pivot2.shape
print(pivot2.iloc[:,6])#total de suscriptores por tipo para cada terminal


#Duracion promedio de los viajes por tipo de suscriptor 
data[['Subscriber Type', 'Duration']].groupby(['Subscriber Type'], as_index = False).mean()
sns.factorplot(x='Subscriber Type', y ='Duration', data=data, kind="bar", size=3)
plt.show()


#Histogramas con distribucion de las variables:
sns.distplot(data['Duration'])
plt.show()
sns.distplot(data['Bike #'])
plt.show()
sns.distplot(data['Start Station'])
plt.show()
#%%
%matplotlib inline
#Matriz de Correlacion lineal
data.corr()
#heatmap de las correlaciones
plt.figure(figsize=(8,4))
sns.heatmap(datos.corr(),cmap='Greens',annot=False)
#matriz de scatter plots para todas la variables, se puede hacer por grupos o individual
scatter_matrix(data)
#%%
#Gragica de los suscriptores y terminales  por fecha

#creamos otro dataframe con las columnas de fechas y terminales
time_data=data.iloc[:,[2,3,4,5,6,7,9]]#dataset de fechas, estaciones y suscriptores
list(time_data.columns)
time_data.head()
#Ahora aplicamos la transformacion de formsto de fecha

#ejemplo de como transformar strings a date time
fecha = pd.to_datetime(data['Start Date'])#correcta transformacion de fechas
print(fecha)

#fecha vs suscriptores
time_data=data.sort_values('Start Date', ascending=True)
plt.plot(time_data['Start Date'], time_data['Subscriber Type'])
plt.xticks(rotation='vertical')

#fecha vs start terminal
time_data=time_data.sort_values('Start Date', ascending=True)
plt.plot(time_data['Start Date'], time_data['Start Terminal'])
plt.xticks(rotation='vertical')

#%%

"""
Para crear las graficas de la duracion min, promedio y maxima de los tiempos por mes y de
acuerdo a los tipos de suscriptores, primero distinguimos valores por mes, después, ya 
para graficar debemos separar las duraciones minimas y medias en una grafica  y las 
duraciones maximas en otra ya que los valores al ser muy distintos; al visualizacion no
se alcanzan a distinguir adecuadamente
"""
#Transformas la data a tipo datetime
data['Start Date']=pd.to_datetime(data['Start Date'])
#creamos una variable que devuelva el mes
data['Month']=data['Start Date'].apply(lambda x: x.strftime('%B'))
#observamos la nueva columna
list(data.columns)
data.head()
data.tail()
#Veamos la distribucion de los meses
data['Month'].unique()
data['Month'].value_counts()
#Ahor avamos a filtrar el dataset por meses, y se puede analizar por mes
March=data[data['Month']=='March']
April=data[data['Month']=='April']
May=data[data['Month']=='May']
June=data[data['Month']=='June']
July=data[data['Month']=='July']
August=data[data['Month']=='August'] 
#Calculamos el vector de medias y los valores minimos y maximos para todas la variables:
means = data.groupby('Month').mean()
type(means)
means = pd.DataFrame(means)#transformamos a dataframe
means
means['Month']=means.index
means
#Ls minimos
mini = data.groupby('Month').min()
mini = pd.DataFrame(mini)#transformamos a dataframe
mini['Month']=mini.index
#Ahora los maximos
maxi = data.groupby('Month').max()
maxi = pd.DataFrame(maxi)#transformamos a dataframe
maxi['Month']=maxi.index

#Agrupamo slos resultados en un solo dataset (el de lo sminimos)
mini['Maxi']=maxi['Duration']#le anadimos las columnas
mini['Means']=means['Duration']
mini.shape
values=mini.iloc[:,[0,9,10,11]]#filtramos
values.rename(columns = {'Duration': 'Min'}, inplace=True)#renombramos
values=values.reset_index(drop= True)#reindexamos
print(values.columns)
values

#Para graficar vamos a separar min con media y max por separado
min_mean=values.iloc[:,[0,1,3]]

min_mean.plot(x='Month', kind='bar',stacked=False,
            title='Duration Min & Mean by month')
#Los maximos
values.iloc[:,[1,2]].plot(x='Month', kind='bar',stacked=False,
            title='Max duration by month')

#######AHORA GRAFICAMOS LAS DURACIONES DE ACUERDO A LAS PREFENCIAS DE LOS SUCRIPTORES
sus_type=data.iloc[:,[1,9]]
sus_type.head()

means_s = sus_type.groupby('Subscriber Type').mean()
means_s = pd.DataFrame(means_s)#transformamos a dataframe
means_s
means_s['Subscriber Type']=means_s.index
means_s
#Los minimos
mini_s = sus_type.groupby('Subscriber Type').min()
mini_s = pd.DataFrame(mini_s)#transformamos a dataframe
mini_s['Subscriber Type']=mini_s.index
#Ahora los maximos
maxi_s = sus_type.groupby('Subscriber Type').max()
maxi_s = pd.DataFrame(maxi_s)#transformamos a dataframe
maxi_s['Month']=maxi_s.index

#Agrupamo slos resultados en un solo dataset (el de lo sminimos)
mini_s['Maxi']=maxi_s['Duration']#le anadimos las columnas
mini_s['Means']=means_s['Duration']
mini_s.shape
mini_s
values_s=mini_s
values_s
values_s.rename(columns = {'Duration': 'Min'}, inplace=True)#renombramos
values_s=values_s.reset_index(drop= True)#reindexamos
print(values_s.columns)
values_s

#Para graficar vamos a separar min con media y max por separado

values_s.iloc[:,[0,1,3]].plot(x='Subscriber Type', kind='bar',stacked=False,
            title='Duration Min & Mean by suscriber type')
#Los maximos
values_s.iloc[:,[1,2]].plot(x='Subscriber Type', kind='bar',stacked=False,
            title='Max duration by suscriber type')

#%%
#MValores faltantes:
sns.heatmap(datos.isnull(), cbar=False, yticklabels=False, cmap='viridis')

from dateutil.parser import parse, parser
import datetime
parse('03/01/2014  12:16:00 a. m.').date()
print(datetime.date(2014, 3, 1))



df['A'] = df['A'].apply(add_2)
print (df)
time_data['Start Date'] = time_data['Start Date'].apply(parse)
#time_data['Start Date'] = time_data['Start Date'].apply(date())
print (time_data)

type(time_data['Start Date'])

def string_date_time(dstr):
    d = parser.parse(dstr)
    d= d.strftime("%Y-%m-%d")
    return d

string_date_time(time_data['Start Date'])

###
import datetime as dt

datetime.datetime.strptime
Enero=data[(data['Start Date']>=datetime.datetime.strptimedate(2014,1,1)) &
           (data['Start Date']<=datetime.datetime.strptime.date(2014,1,31))] 

Enero=data[(data['Start Date']>=datetime.date(2014,1,1)) &
           (data['Start Date']<=datetime.date(2014,1,31))] 

type(data['Start Date'])

dt.datetime.date(2014,1,1)

dt.datetime.date(2014,1,1)
dt.datetime.strptime()

data_fecha = data.set_index('Start Date')
data_fecha.head()

# Se crean las fechas con la librería datetime
ene_inic = dt.datetime(2014, 1, 1)
ene_fin = dt.datetime(2014, 1, 31)
# Filtro por fecha
data_fecha.loc[ene_inic: ene_fin]


date_string = "12/11/2018"
date_object = dt.strptime(date_string, "%d %m %Y")

print("date_object =", date_object)


###

###

#Vamos a graficar por mes las medias y los valores minimos y maximos:
rango = np.arange(6) 
width = 0.2
  # plot data in grouped manner of bar type 
plt.bar(rango-0.2, mini['Duration'], width, color='cyan') 
plt.bar(rango, means['Duration'], width, color='orange') 
plt.bar(rango+0.2, maxi['Duration'], width, color='green') 
plt.xticks(rango, ['March', 'April', 'May', 'June', 'July', 'August']) 
plt.xlabel("Months") 
plt.ylabel("Values") 
plt.legend(["Min", "Mean", "Max"]) 
plt.show()

###
#para la grafica:

# create data 
df = pd.DataFrame([['A', 10, 20, 10, 30], ['B', 20, 25, 15, 25], ['C', 12, 15, 19, 6], 
                   ['D', 10, 29, 13, 19]], 
                  columns=['Team', 'Round 1', 'Round 2', 'Round 3', 'Round 4']) 
# view data 
print(df) 
  
# plot grouped bar chart 
df.plot(x='Team', 
        kind='bar', 
        stacked=False, 
        title='Grouped Bar Graph with dataframe')

"""
Para crear las graficas de la duracion min, promedio y maxima de los tiempos por mes y de
acuerdo a los tipos de suscriptores, primero distinguimos valores por mes, después, ya 
para graficar debemos separar las duraciones minimas y medias en una grafica  y las 
duraciones maximas en otra ya que los valores al ser muy distintos; al visualizacion no
se alcanzan a distinguir adecuadamente
"""
#Transformas la data a tipo datetime
data['Start Date']=pd.to_datetime(data['Start Date'])
#creamos una variable que devuelva el mes
data['Month']=data['Start Date'].apply(lambda x: x.strftime('%B'))
#observamos la nueva columna
list(data.columns)
data.head()
data.tail()
#Veamos la distribucion de los meses
data['Month'].unique()
data['Month'].value_counts()
#Ahor avamos a filtrar el dataset por meses, y se puede analizar por mes
March=data[data['Month']=='March']
April=data[data['Month']=='April']
May=data[data['Month']=='May']
June=data[data['Month']=='June']
July=data[data['Month']=='July']
August=data[data['Month']=='August'] 
#Calculamos el vector de medias y los valores minimos y maximos para todas la variables:
means = data.groupby('Month').mean()
type(means)
means = pd.DataFrame(means)#transformamos a dataframe
means
means['Month']=means.index
means
#Ls minimos
mini = data.groupby('Month').min()
mini = pd.DataFrame(mini)#transformamos a dataframe
mini['Month']=mini.index
#Ahora los maximos
maxi = data.groupby('Month').max()
maxi = pd.DataFrame(maxi)#transformamos a dataframe
maxi['Month']=maxi.index

#Agrupamo slos resultados en un solo dataset (el de lo sminimos)
mini['Maxi']=maxi['Duration']#le anadimos las columnas
mini['Means']=means['Duration']
mini.shape
values=mini.iloc[:,[0,9,10,11]]#filtramos
values.rename(columns = {'Duration': 'Min'}, inplace=True)#renombramos
values=values.reset_index(drop= True)#reindexamos
print(values.columns)
values

#Para graficar vamos a separar min con media y max por separado
min_mean=values.iloc[:,[0,1,3]]

min_mean.plot(x='Month', kind='bar',stacked=False,
            title='Duration Min & Mean by month')
#Los maximos
values.iloc[:,[1,2]].plot(x='Month', kind='bar',stacked=False,
            title='Max duration by month')

#######AHORA GRAFICAMOS LAS DURACIONES DE ACUERDO A LAS PREFENCIAS DE LOS SUCRIPTORES
sus_type=data.iloc[:,[1,9]]
sus_type.head()

means_s = sus_type.groupby('Subscriber Type').mean()
means_s = pd.DataFrame(means_s)#transformamos a dataframe
means_s
means_s['Subscriber Type']=means_s.index
means_s
#Los minimos
mini_s = sus_type.groupby('Subscriber Type').min()
mini_s = pd.DataFrame(mini_s)#transformamos a dataframe
mini_s['Subscriber Type']=mini_s.index
#Ahora los maximos
maxi_s = sus_type.groupby('Subscriber Type').max()
maxi_s = pd.DataFrame(maxi_s)#transformamos a dataframe
maxi_s['Month']=maxi_s.index

#Agrupamo slos resultados en un solo dataset (el de lo sminimos)
mini_s['Maxi']=maxi_s['Duration']#le anadimos las columnas
mini_s['Means']=means_s['Duration']
mini_s.shape
mini_s
values_s=mini_s
values_s
values_s.rename(columns = {'Duration': 'Min'}, inplace=True)#renombramos
values_s=values_s.reset_index(drop= True)#reindexamos
print(values_s.columns)
values_s

#Para graficar vamos a separar min con media y max por separado

values_s.iloc[:,[0,1,3]].plot(x='Subscriber Type', kind='bar',stacked=False,
            title='Duration Min & Mean by suscriber type')
#Los maximos
values_s.iloc[:,[1,2]].plot(x='Subscriber Type', kind='bar',stacked=False,
            title='Max duration by suscriber type')