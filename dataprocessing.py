import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import loaddata
import math
from scipy import stats
from datetime import datetime, timedelta

def MPE(x,xref):
    n=x.shape[0]
    MPE=1/n*(np.sum(np.absolute(x-xref)/x))*100
    return MPE

def data_visualisation(data,normierung=0,mean=False,std=False,allinone=False):
    if allinone:
        for season in range(0,3):
            for day in range(0,3):
                df=data.loc[:,season].loc[:,day]
                df=df.mean(axis=1)
                plt.plot(df.to_numpy())
    else:
        if mean:
            subfig, axs = plt.subplots(2,3,figsize=(20,10))
        else:
            subfig, axs = plt.subplots(3,3,figsize=(20,15))
        for season in range(0,3):
            for day in range(0,3):
                df=data.loc[:,season].loc[:,day]
                if normierung == 1:
                    df=df/df.mean()
                if normierung == 2:
                    df=(df-df.min())/(df.max()-df.min())
                if mean:
                    df=df.mean(axis=1)
                    axs[0,season].plot(df.to_numpy())
                    axs[0,season].set_title('Jahreszeit ' + str(season))
                    axs[0,season].set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))
                    axs[1,day].plot(df.to_numpy())
                    axs[1,day].set_title('Wochentag ' + str(day))
                    axs[1,day].set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))
                if std:
                    df=df.std(axis=1)
                    axs[0,season].plot(df.to_numpy())
                    axs[0,season].set_title('Jahreszeit ' + str(season))
                    axs[1,day].plot(df.to_numpy())
                    axs[1,day].set_title('Wochentag ' + str(day))

def normalize_dataframe(df,axis=0,global_norm=False):
    if isinstance(df,np.ndarray):
        if global_norm:
            df = (df-df.min().min())/(df.max().max()-df.min().min())
        else:
            if axis == 0:
                df = (df-np.min(df))/(np.max(df)-np.min(df))
            elif axis == 1:
                df = (df-np.min(df,axis=1))/(np.max(df,axis=1)-np.min(df,axis=1))
            df = (df-df.min())/(df.max()-df.min())
    elif isinstance(df,pd.DataFrame):
        if global_norm:
            df = (df-df.min())/(df.max()-df.min())
        else:
            if axis == 0:
                df = (df-df.min())/(df.max()-df.min())
            elif axis == 1:
                df = (df-df.min())/(df.max()-df.min())
    return df

def data_eliminate_zero(df):
    for i in range(1,df.shape[1]):
        last_neq0=0
        first_neq0=0
        for j in range(0,df.shape[0]):
            if df.iloc[:,i].iloc[j]>0:
                if last_neq0 == first_neq0:
                    last_neq0=last_neq0+1
                    first_neq0=first_neq0+1
                else:
                    first_neq0=first_neq0+1
                    ersatzwert=1/2*(df.iloc[:,i].iloc[last_neq0-1]+df.iloc[:,i].iloc[first_neq0-1])
                    for k in range(0,first_neq0-last_neq0-1):
                        df.iloc[:,i].iloc[last_neq0+k]=ersatzwert
                    last_neq0=first_neq0
            else:
                first_neq0=first_neq0+1
    return df

def interpolate_data(df):
    n=df.shape[0]
    for i in range(0,n):
        for j in range(1,4):
            if df.iloc[i,j]<= 0:
                df.iloc[i,j]=0
    
    df['A-1-träge-max (L1)']=df['A-1-träge-max (L1)'].replace(0,np.nan)
    df['A-2-träge-max (L2)']=df['A-2-träge-max (L2)'].replace(0,np.nan)
    df['A-3-träge-max (L3)']=df['A-3-träge-max (L3)'].replace(0,np.nan)

    df['A-1-träge-max (L1)']=df['A-1-träge-max (L1)'].interpolate()
    df['A-2-träge-max (L2)']=df['A-2-träge-max (L2)'].interpolate()
    df['A-3-träge-max (L3)']=df['A-3-träge-max (L3)'].interpolate()

    return df

def pivot_data(df,bool_year,bool_season,bool_month,bool_weekday,anonomize_date):
    #Pivotisierung der Daten
    ###INPUT:
    #df: DataFrame, das pivotisiert werden soll
    ###OUTPUT:
    #df_pivot: DataFrame, nach Pivotisierung

    n = df.shape[0]

    #Fuege Wochentag und Monat hinzu.
    df["Monat"]=df["Datum/Uhrzeit"].dt.month
    df['Monat']=df['Monat'].astype(int)
    df["Jahr"]=df["Datum/Uhrzeit"].dt.year
    
    ###DataFrame auf vollstaendige Tage begrenzen
    #Suche ersten vollstaendigen Tag
    i=0
    while i<=n and (df["Datum/Uhrzeit"][i].hour != 0 or df["Datum/Uhrzeit"][i].minute > 15):
        i = i+1
    start = i

    #Suche letzten vollstaendigen Tag
    i=n-1
    while i>=0 and (df["Datum/Uhrzeit"][i].hour != 23 or df["Datum/Uhrzeit"][i].minute < 45):
        i = i-1
    ende = i

    df_mod=df[start:ende+1]

    #Einheitliche 15 Minuten Abtastzeiten    
    df_mod=df_mod.resample('15T',on='Datum/Uhrzeit').mean()
    df_mod=df_mod.reset_index()

    df_mod["Datum"]=df_mod["Datum/Uhrzeit"].dt.date
    df_mod["Zeit"]=df_mod["Datum/Uhrzeit"].dt.time

    df_mod["Datum"]=pd.to_datetime(df_mod['Datum']).dt.strftime("%Y-%m-%d")
    
    jahresdaten = pd.read_excel('resources/jahr171819_final.xlsx')
    jahresdaten['Datum']=jahresdaten['Datum'].dt.strftime("%Y-%m-%d")
    #print(type(jahresdaten['Datum']))
    jahresdaten=jahresdaten.set_index('Datum')
    
    df_mod=df_mod.join(jahresdaten,on='Datum')
    if anonomize_date:
        df_mod["Datum"]=pd.to_datetime(df_mod['Datum']).dt.strftime("%j")
    
    df_mod['Strom']=df_mod['A-1-träge-max (L1)']+df_mod['A-2-träge-max (L2)']+df_mod['A-3-träge-max (L3)']          #Strom der 3 Phasen summieren

    df_mod['Leistung']=df_mod['Strom']*400*1/(math.sqrt(3)*1e3)                                                     #Leistung in kW berechnen
    
    col=list()
    if bool_year:
        col.insert(len(col),'Jahr')
    if bool_season:
        col.insert(len(col),'Jahreszeit')
    if bool_month:
        col.insert(len(col),'Monat')
    if bool_weekday:
        col.insert(len(col),'Wochentag')
    col.insert(len(col),'Datum')
    df_pivot = pd.pivot_table(df_mod,values='Leistung',index='Zeit',columns=col,aggfunc=np.mean)
        
    return df_pivot

def norm_data_for_simulation(data_in, case_standardisation):
    #Normierung von Daten gemaess eingestellter Normierungsoptionen
    ###INPUT:
    #data: DataFrame
    #case_standardisation: Liste mit Nomierungsoptionen mit
    #   Erste Stelle:   0: Keine Normierung, 1: Normierung auf Mittelwert, 2: Normierung auf Min/Max, 3: z-Standardisierung
    #   Zweite Stelle:  0: Lokale Normierung, 1: Globale Normierung
    #   Dritte Stelle:  0: Feste Grenzen, 1: Dynamische Grenzen
    #   Vierte Stelle:  0: Faktor, 1: Absolut
    ###OUTPUT:
    #data_for_simulation: pd.DataFrame - Normierte Daten
    #extreme_values_data: array - Globale Extremwerte [min_wert, mean_wert, max_wert]
        
    #Normierungsoptionen aufschluesseln
    case_1 = case_standardisation[0]
    case_2 = case_standardisation[1]
    case_3 = case_standardisation[2]
    case_4 = case_standardisation[3]

    if isinstance(data_in,np.ndarray):
        datamod=pd.DataFrame(data_in)
    else:
        datamod=data_in

    data=datamod.copy()

    #Globale Extremwerte bestimmen
    max_wert = data.max().max()
    mean_wert = data.mean().mean()
    min_wert = data.min().min()

    print('Normiere Daten gemaess: Normierung:', case_1, ', Lokal/Global:', case_2, ', Grenzen:', case_3, ', Reskalierung:', case_4)

    #Daten normieren
    if case_1 == 1:
        if case_2 == 0:
            data_for_simulation = data/data.mean() 
        else:
            data_for_simulation = data/mean_wert
    elif case_1 == 2:
        if case_2 == 0:
            data_for_simulation = (data-data.min())/(data.max()-data.min())
        else:
            data_for_simulation = (data-min_wert)/(max_wert-min_wert)
    elif case_1 == 3:
        data_for_simulation = stats.zscore(data)
    elif case_1 == 4:
        if case_2==0:
            data_for_simulation = data/data.max()
        elif case_2==2:
            data_for_simulation = data.copy()
            season = 0
            data_season = data.loc[:,season]
            n0=data_season.shape[1]
            max_wert_season = data_season.max().max()
            for j in range(0,n0):
                data_for_simulation.iloc[:,j] = data.iloc[:,j]/max_wert_season
            season = 1
            data_season = data.loc[:,season]
            n1=data_season.shape[1]
            max_wert_season = data_season.max().max()
            for j in range(n0,n0+n1):
                data_for_simulation.iloc[:,j] = data.iloc[:,j]/max_wert_season
            season = 2
            data_season = data.loc[:,season]
            n2=data_season.shape[1]
            max_wert_season = data_season.max().max()
            for j in range(n0+n1,n0+n1+n2):
                data_for_simulation.iloc[:,j] = data.iloc[:,j]/max_wert_season
        else:
            data_for_simulation = data/max_wert
    else:
        data_for_simulation=data
    extreme_values_data = [min_wert, mean_wert, max_wert]

    return data_for_simulation, extreme_values_data

def normalize_by_week(df_mat):
    #Berechnung einer Normierung fuer eine Woche
    ###INPUT:
    #df_mat: 96x9-Matrix
    ###OUTPUT:
    #normed_mat: Normierte 96x9-Matrix

    #Initialisierung einer Matrix
    normed_mat = {}

    #Fuer jede Jahreszeit normieren
    for season in range(0,3):
        for weekday in range(0,3):
            normed_mat[season,weekday]=df_mat[season,weekday].mean(axis=1)

        max_season = np.max([normed_mat[season,0].max(),normed_mat[season,1].max(),normed_mat[season,2].max()])             #Berechnung des Maximums der Woche
        
        for weekday in range(0,3):
            normed_mat[season,weekday]=normed_mat[season,weekday]/max_season                                                #Normierung auf Wochenmaximum
   
    return normed_mat

def stochastic_analysis(df,varianzkoeff=False):
    ###Plotten von Mittelwert und Standardabweichung im Tagesverlauf
    #INPUT
    #df: DataFrame mit Spalten Jahreszeit, Wochentag, Datum

    dict_season, dict_weekday = loaddata.dicts_season_weekday()
    
    if varianzkoeff:
        subfig, axs = plt.subplots(3, 1,figsize=(6.8,14.4))
    else:
        subfig, axs = plt.subplots(2, 1,figsize=(24,9.6))

    max_mean = np.zeros(9)
    max_std = np.zeros(9)
    min_mean = np.zeros(9)
    min_std = np.zeros(9)
    max_vk = np.zeros(9)
    min_vk = np.zeros(9)
    count = 0
    vk_mat=np.zeros([3,3])
    
    for weekday in range(0,1):
        for season in range(0,3):
            this_df=df.loc[:,season].loc[:,weekday]
            this_df_mean=this_df.mean(axis=1).to_numpy()
            this_df_std=this_df.std(axis=1).to_numpy()
            this_df_factor=this_df_std/this_df_mean
            max_mean[count]=np.max(this_df_mean)
            min_mean[count]=np.min(this_df_mean)
            max_std[count]=np.max(this_df_std)
            min_std[count]=np.min(this_df_std)
            max_vk[count]=np.max(this_df_factor)
            min_vk[count]=np.min(this_df_factor)
            
            axs[0].plot(this_df_mean)
            axs[0].set_title('Mittelwert im Tagesverlauf am Tag ' + dict_weekday[weekday])
            axs[0].legend(['Winter','Übergangszeit','Sommer'],bbox_to_anchor=(1,1))
            axs[0].set_xlabel('Zeit [h]')
            axs[0].set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))
            axs[0].set_ylabel('Leistung [kW]')
            axs[0].grid()
            axs[1].plot(this_df_std)
            axs[1].set_title('Standardabweichung im Tagesverlauf am Tag ' + dict_weekday[weekday])
            axs[1].legend(['Winter','Übergangszeit','Sommer'],bbox_to_anchor=(1,1))
            axs[1].set_ylabel('Leistung [kW]')
            axs[1].set_xlabel('Zeit [h]')
            axs[1].set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))
            axs[1].grid()
            if varianzkoeff:
                axs[2].plot(this_df_factor)
                axs[2].set_title('Variationskoeffizient im Tagesverlauf am Tag ' + dict_weekday[weekday])
                axs[2].legend(['Winter','Übergangszeit','Sommer'],bbox_to_anchor=(1,1))
                axs[2].set_ylabel('Variationskoeffizient')
                axs[2].set_xlabel('Zeit [h]')
                axs[2].set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))
                axs[2].grid()
            count=count+1
            vk_mat[season,weekday]=np.mean(this_df_factor)
    for weekday in range(0,1):
        axs[0].set_ylim([np.min(min_mean)*0.9,np.max(max_mean)*1.1])
        axs[1].set_ylim([np.min(min_std)*0.9,np.max(max_std)*1.1])
        if varianzkoeff:
            axs[2].set_ylim([np.min(min_vk)*0.9,np.max(max_vk)*1.1])
    plt.subplots_adjust(hspace=0.5)
    return vk_mat