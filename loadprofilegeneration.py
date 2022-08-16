import numpy as np
import pandas as pd
import dataprocessing as dp
from loaddata import dicts_season_weekday
from dataprocessing import norm_data_for_simulation
from loadmodelingfunctions import calc_random_gmm_bins
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tslearn.clustering import KShape
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import norm, zscore
from scipy import stats
from numpy.fft import fft
import matplotlib.mlab as mlab
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def generate_loadprofile(df_last_matrix,df_last_oneyear,export=False,name='',plots=False,deg=4):
    #Erstellung von statischen Lastprofilen
    ###INPUT:
    #df_last_matrix: Lastprofile geordnet nach Typtagen (Jahreszeit, Wochentag)
    #df_last_oneyear: Lastdaten mit Jahresaufloesung
    #export: Boolean - Export als CSV
    #name: Dateiname fuer den Export
    #plots: Boolean - Plotten
    #deg: Grad des Regressionspolynoms

    year = 2018

    #Berechnung des Regressionspolynoms
    coeff_regression_energy = approx_energy(df_last_oneyear.loc[:,year],deg)

    #Normierung
    normed_load_matrix = normalize_by_week(df_last_matrix)

    for season in range(0,3):
        for weekday in range(0,3):
            dayenergy=0.25*normed_load_matrix[season,weekday].sum()
            normed_load_matrix[season,weekday]=normed_load_matrix[season,weekday]/dayenergy

    load_example_year=build_example_year(normed_load_matrix,coeff_regression_energy)

    load_example_year_wide = np.reshape(load_example_year,[96,365],order='F')
    plt.figure()
    plt.plot(np.mean(load_example_year_wide,axis=0))

    df_example_year=pd.DataFrame(load_example_year)                     #in kW
    jahresenergie=0.9*calc_energy(df_example_year)                      #in kWh
    if plots:
        df_example_year.plot(legend=False)
        plt.figure()
        plt.title('Beispieljahr im Jahresverlauf')
        plt.ylabel('Normierte Leistung')
        plt.xlabel('Zeitschritt im Jahr')
    
    load_profiles_numpy = np.zeros([96,9])
    for season in range(0,3):
        for weekday in range(0,3):
            load_profiles_numpy[:,3*season+weekday]=normed_load_matrix[season,weekday]/(jahresenergie/1000)

    iterables=[['Winter','Übergangszeit','Sommer'],['Werktag','Samstag','Sonntag']]
    ind=pd.MultiIndex.from_product(iterables, names=['Season','Weekday'])

    load_profiles = pd.DataFrame(load_profiles_numpy, columns=ind)
    load_profiles.plot()

    if export:
        load_profiles.to_csv('load_profiles_' + name + '.csv')
    
    return load_profiles, coeff_regression_energy

def approx_energy(load,deg,deltaT=0.25,plots=True):
    #Berechnung des Regressionspolynoms fuer die Energie
    ###INPUT:
    #load: DataFrame oder Array eindimensional
    #deg: Grad des Regressionspolynoms
    #deltaT, plots
    ###OUTPUT:
    #reg_poly: Koeffizienten des Regressionspolynoms

    if isinstance(load,pd.DataFrame):
        load=load.to_numpy()
    
    Y=deltaT*np.sum(load)
    n=Y.shape[0]                #Laenge der Lastdaten

    Yfitting = Y / Y.mean()

    X=np.linspace(1,n,n)
    reg_poly_coeff=np.polyfit(X,Yfitting,deg)
    reg_poly=np.poly1d(reg_poly_coeff)

    if plots:
        plt.figure()
        plt.scatter(X,Yfitting)
        plt.plot(reg_poly(X),color='red')
        plt.title('Energieverbrauch pro Tag')
        plt.xlabel('Tag im Jahr')
        plt.ylabel('Normierte Energie')
        #plt.savefig('Jahresenergie.pdf',bbox_inches='tight')
    return reg_poly

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

def build_example_year(load_profile,coeff_regression_energy):
    #Erstellung eines Beispieljahres
    ###INPUT:
    #load_profile: 96x9 Matrix mit Lastprofilen (Jahreszeit, Wochentag)
    #coeff_regression_energy: Regressionspolynom der Energie
    ###OUTPUT:
    #load_example_year: Jahres-Lastgang der Laenge 96*365=35040
    
    #Importiere Zuordnung von Wochentagen und Jahreszeiten
    example_year = pd.read_excel('bspjahr.xlsx')

    #Importiere Dictionary
    season_dict, weekday_dict = hf.dicts_season_weekday()
    
    #Initialisierung
    load_example_year = np.zeros([35040,1])

    #Berechnung der approximierten Tagesenergien
    days_of_the_year = np.linspace(1,365,365)
    factors_of_the_year = np.poly1d(coeff_regression_energy)(days_of_the_year)

    #Befuellung des Beispieljahres mit Lastprofilen
    if isinstance(load_profile,dict):                                                              #load_profile ist dict
        for i in range(0,365):
            weekday = example_year['Tag'].iloc[i]
            season = example_year['Jahreszeit'].iloc[i]
            load_example_year[i*96:(i+1)*96,0]=factors_of_the_year[i]*load_profile[season,weekday]
    elif isinstance(load_profile,pd.DataFrame):                                                    #load_profile ist 96x9 DataFrame
        for i in range(0,365):                                          
            weekday = weekday_dict[example_year['Tag'].iloc[i]]
            season = season_dict[example_year['Jahreszeit'].iloc[i]]
            load_example_year[i*96:(i+1)*96,0]=factors_of_the_year[i]*load_profile.loc[:,season].loc[:,weekday]

    energy = calc_energy(load_example_year)
    print(energy)

    return load_example_year

def stochastic_analysis(df,varianzkoeff=False):
    ###Plotten von Mittelwert und Standardabweichung im Tagesverlauf
    #INPUT
    #df: DataFrame mit Spalten Jahreszeit, Wochentag, Datum

    dict_season, dict_weekday = hf.dicts_season_weekday()
    """
    for season in range(0,3):
        plt.figure()
        subfig, axs = plt.subplots(2, 3,figsize=(24,9.6))
        for weekday in range(0,3):
            this_df=df.loc[:,season].loc[:,weekday]
            this_df_mean=this_df.mean(axis=1).to_numpy()
            this_df_std=this_df.std(axis=1).to_numpy()
            axs[0,weekday].plot(this_df_mean)
            axs[0,weekday].set_title('Mittelwert - Jahreszeit ' + dict_season[season] + ', Wochentag ' + dict_weekday[weekday])
            axs[1,weekday].plot(this_df_std)
            axs[1,weekday].set_title('Standardabweichung - Jahreszeit ' + dict_season[season] + ', Wochentag ' + dict_weekday[weekday])
    """
    if varianzkoeff:
        subfig, axs = plt.subplots(3, 3,figsize=(24,14.4))
    else:
        subfig, axs = plt.subplots(2, 3,figsize=(24,9.6))

    max_mean = np.zeros(9)
    max_std = np.zeros(9)
    min_mean = np.zeros(9)
    min_std = np.zeros(9)
    max_vk = np.zeros(9)
    min_vk = np.zeros(9)
    count = 0
    vk_mat=np.zeros([3,3])
    for weekday in range(0,3):
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
            axs[0,weekday].plot(this_df_mean)
            axs[0,weekday].set_title('Mittelwert im Tagesverlauf am Tag ' + dict_weekday[weekday])
            axs[0,weekday].legend(['Winter','Übergangszeit','Sommer'])
            axs[0,weekday].set_xlabel('Zeit [h]')
            axs[0,weekday].set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))
            axs[0,weekday].set_ylabel('Leistung [kW]')
            axs[0,weekday].grid()
            axs[1,weekday].plot(this_df_std)
            axs[1,weekday].set_title('Standardabweichung im Tagesverlauf am Tag ' + dict_weekday[weekday])
            axs[1,weekday].legend(['Winter','Übergangszeit','Sommer'])
            axs[1,weekday].set_ylabel('Leistung [kW]')
            axs[1,weekday].set_xlabel('Zeit [h]')
            axs[1,weekday].set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))
            axs[1,weekday].grid()
            if varianzkoeff:
                axs[2,weekday].plot(this_df_factor)
                axs[2,weekday].set_title('Variationskoeffizient im Tagesverlauf am Tag ' + dict_weekday[weekday])
                axs[2,weekday].legend(['Winter','Übergangszeit','Sommer'])
                axs[2,weekday].set_ylabel('Variationskoeffizient')
                axs[2,weekday].set_xlabel('Zeit [h]')
                axs[2,weekday].set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))
                axs[2,weekday].grid()
            count=count+1
            vk_mat[season,weekday]=np.mean(this_df_factor)
    for weekday in range(0,3):
        axs[0,weekday].set_ylim([np.min(min_mean)*0.9,np.max(max_mean)*1.1])
        axs[1,weekday].set_ylim([np.min(min_std)*0.9,np.max(max_std)*1.1])
        if varianzkoeff:
            axs[2,weekday].set_ylim([np.min(min_vk)*0.9,np.max(max_vk)*1.1])
    plt.subplots_adjust(hspace=0.3)

    return vk_mat

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    set_link_color_palette(['g', 'r', 'c', 'm', 'y', 'k'])
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix,**kwargs)