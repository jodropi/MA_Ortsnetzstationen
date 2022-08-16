import numpy as np
import pandas as pd
import dataprocessing as dp
import loaddata
from dataprocessing import norm_data_for_simulation
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

def calc_energy(df):
    E=0.25*df.sum(axis=0).sum(axis=0)
    return E

def calc_daily_energy(daily_leistung):
    daily_energy=np.zeros(365)
    for day in range(0,365):
        daily_energy[day]=0.25*np.sum(daily_leistung[96*day:96*(day+1)])
    year_energy=np.sum(daily_energy)
    return daily_energy, year_energy

def calc_regression_polynoms(data_in,data_weekday_in,deg,case_normierung,plots=False,exact_values=False):
    #Berechnung des Regressions-Polynoms der Energie
    ###INPUT:
    #data_in:           array-like - Daten
    #deg:               int - Grad des Regressions-Polynoms
    #case_normierung:   list
    ###OUTPUT:
    #energy_per_year:   int - Energie pro Jahr der Daten
    #c_factors:         array - Faktoren zur Skalierung des Mittelwerts [c_min, c_mean, c_max]
    #std_factors:       ...
    #extreme_vec:       list - Vektoren der täglichen Minimal-, Mittel- und Maximalwerte [min_vec, mean_vec, max_vec]
    #std_vec:           ...

    #Erste Stelle: 0: Keine Normierung, 1: Normierung auf Mittelwert, 2: Normierung auf Min/Max, 3: z-Standardisierung
    #Zweite Stelle: 0: Lokale Normierung, 1: Globale Normierung
    #Dritte Stelle: 0: Feste Grenzen, 1: Dynamische Grenzen
    #Vierte Stelle: 0: Faktor, 1: Absolut
    case_1 = case_normierung[0]
    case_2 = case_normierung[1]
    case_3 = case_normierung[2]
    case_4 = case_normierung[3]

    if isinstance(data_in, np.ndarray):
        data=pd.DataFrame(data_in)
    else:
        data=data_in

    #display(data)

    if isinstance(data_weekday_in, np.ndarray):
        data_w=pd.DataFrame(data_weekday_in)
    else:
        data_w=data_weekday_in

    energy_per_year=calc_energy(data)
    
    X=np.linspace(0,364,365)                        #Tage im Jahr

    Ymin=data.min().to_numpy()
    Ymean=data.mean().to_numpy()
    Ystd=data.std().to_numpy()
    Ymax=data.max().to_numpy()

    true_values = [Ymin, Ymean, Ymax]

    reg_poly_max=np.polyfit(X,Ymax,deg)
    mymodel_max=np.poly1d(reg_poly_max)


    #Ymaxmean=np.mean(Ymax)
    #Yminmean=np.mean(Ymax)
    #Ymin=data.min().to_numpy()/data.min().min()
    #Ymean=data.mean().to_numpy()/data.mean().mean()
    #Ymax=data.max().to_numpy()/data.max().max()
    
    #Y_faktor_max=data.max().to_numpy()/data.mean().to_numpy()
    #Y_faktor_min=data.min().to_numpy()/data.mean().to_numpy()

    min_vec  = Ymin
    mean_vec = Ymean
    #max_vec  = Ymax
    max_vec = mymodel_max(X)
    std_vec  = Ystd

    diff_max=Ymax-max_vec
    std_max=np.std(diff_max)

    Y_faktor_max=Ymax/Ymean
    Y_faktor_min=Ymin/Ymean

    if case_4==0:
        max_vec=Y_faktor_max
        min_vec=Y_faktor_min

    """
    plt.figure()
    plt.hist(Y_faktor_max,bins=50)
    plt.figure()
    plt.hist(Y_faktor_min,bins=50)
    plt.figure()
    plt.hist(Y_faktor_delta,bins=50)
    """

    c_max = np.mean(Y_faktor_max)
    c_min = np.mean(Y_faktor_min)
    c_mean = 1

    #std_max = np.std(Y_faktor_max)
    std_min = np.std(Y_faktor_min)
    std_mean = 1

    c_factors = [c_min, c_mean, c_max]
    std_factors = [std_min, std_mean, std_max]
    
    #Regressionspolynom bestimmen
    reg_poly_mean=np.polyfit(X,Ymean,deg)
    mymodel_mean=np.poly1d(reg_poly_mean)

    if case_4==0:
        mean_vec=mymodel_mean(X)/np.mean(mymodel_mean(X))
    elif case_4==1:
        mean_vec=mymodel_mean(X)

    mean_vec=Ymean

    extreme_vec = [min_vec, mean_vec, max_vec]
    
    if plots:
        plt.figure()
        plt.scatter(X,Ymin,color='red')
        plt.scatter(X,Ymean,color='blue')
        plt.scatter(X,Ymax,color='green')
        plt.title('Scatter-Plot der minimalen, mittleren und maximalen Leistung pro Tag')
        
        plt.figure()
        plt.plot(Y_faktor_max)
        #plt.plot(X,mymodel_c_max(X),color='red')
        plt.title('Sklaierungsfaktor c_max=P_max/P_mw')
        plt.suptitle('mean =' + str(np.mean(Y_faktor_max)) + ', std =' + str(np.std(Y_faktor_max)))

        plt.figure()
        plt.plot(Y_faktor_min)
        #plt.plot(X,mymodel_c_min(X),color='red')
        plt.title('Sklaierungsfaktor c_min=P_min/P_mw')
        plt.suptitle('mean =' + str(np.mean(Y_faktor_min)) + ', std =' + str(np.std(Y_faktor_min)))
        
        plt.figure()
        plt.scatter(X,Ymean)
        plt.plot(X,mean_vec,color='red')
        plt.title('Scatter-Plot der mittleren Leistung mit dem Regressions-Polynom pro Tag')

        differenz_mean = mymodel_mean(X)-Ymean
        plt.figure()
        plt.plot(differenz_mean)
        plt.title('Nicht-normierte Differenz von Regressions-Polynom und mittlerer Leistung pro Tag')
        plt.suptitle('std =' + str(np.std(differenz_mean)))


    E_mat = np.zeros([3,3])
    E_mat_std = np.zeros([3,3])
    factors_weekdays = np.zeros([3,3])
    std_weekdays = np.zeros([3,3])
    E_mean_season = np.zeros(3)

    for season in range(0,3):
        for weekday in range(0,3):
            E_mat[season,weekday] = 0.25*data_w.loc[:,season].loc[:,weekday].sum(axis=0).mean()
            E_mat_std[season,weekday] = 0.25*data_w.loc[:,season].loc[:,weekday].sum(axis=0).std()
        E_mean_season[season] = 0.25*data_w.loc[:,season].sum(axis=0).mean()
    
    for season in range(0,3):
        for weekday in range(0,3):
            factors_weekdays[season,weekday] = E_mat[season,weekday]/E_mean_season[season]
            std_weekdays[season,weekday] = E_mat_std[season,weekday]/E_mean_season[season]
    """
    E_0=0.25*data_w.loc[:,0].sum(axis=0).mean()
    E_1=0.25*data_w.loc[:,1].sum(axis=0).mean()
    E_2=0.25*data_w.loc[:,2].sum(axis=0).mean()
    E_mean=0.25*data_w.sum(axis=0).mean()

    factors_0=E_0/E_mean
    factors_1=E_1/E_mean
    factors_2=E_2/E_mean

    factors_weekdays=[factors_0,factors_1,factors_2]
    """

    return energy_per_year, c_factors, extreme_vec, std_factors, std_vec, true_values, factors_weekdays, std_weekdays

def calc_gaussian_mixture(df,n_comp,n_bins=50,plots=False):
    #Bestimmung der GMM
    ###INPUT
    #df:        np.array
    #n_comp:    int - Anzahl der Komponenten
    #n_bins:    int - Anzahl der Grenzen im Histogramm
    #plots:     boolean
    ###OUTPUT:
    #gmm:       GaussianMixtureModel
    #p:         Wahrscheinlichkeiten

    data=df.reshape(-1, 1)      #wide -> long
    gmm = GaussianMixture(n_components=n_comp, covariance_type='full').fit(data)
    p=np.zeros(0)

    if plots:
        plt.figure(figsize=(9.6,4.8))
        _, _, _ = plt.hist(data, bins=n_bins, density=True,color="lightblue")
        xmin, xmax = plt.xlim()
        print(xmin, xmax)
        x = np.linspace(xmin, xmax, 100)
        p=np.zeros(x.shape)
        for j in range(0,n_comp):
            p=p+gmm.weights_[j]*norm.pdf(x, gmm.means_[j][0], np.sqrt(gmm.covariances_[j][0]))
        plt.plot(x, p, 'k', linewidth=2)
        plt.xlabel('Normierte Leistung')
        plt.ylabel('Häufigkeit')
        #plt.xlim([0,1])
        """
        datamax=np.max(data)
        datamin=np.min(data)
        grenzen = np.linspace(datamin,datamax,11)
        print(grenzen)
        for grenze in range(0,11):
            plt.axvline(x=grenzen[grenze],color='r')
            if grenze <=9:
                plt.text((grenzen[grenze+1]-grenzen[grenze])/2+grenzen[grenze], 10, str(grenze+1),horizontalalignment='center')
        #plt.savefig('GMMHistogramm_d.pdf',bbox_inches='tight')
        """
        plt.figure(figsize=(9.6,4.8))
        """
        _, _, _ = plt.hist(data, bins=n_bins, density=True,color="lightblue")
        xmin, xmax = plt.xlim()
        print(xmin, xmax)
        x = np.linspace(xmin, xmax, 100)
        p=np.zeros(x.shape)
        for j in range(0,n_comp):
            p=p+gmm.weights_[j]*norm.pdf(x, gmm.means_[j][0], np.sqrt(gmm.covariances_[j][0]))
        #plt.plot(x, p, 'k', linewidth=2)
        plt.xlabel('Normierte Leistung')
        plt.ylabel('Häufigkeit')
        plt.xlim([0,1])
        grenzen = np.linspace(0,1,11)
        print(grenzen)
        for grenze in range(0,11):
            plt.axvline(x=grenzen[grenze],color='r')
            if grenze <=9:
                plt.text((grenzen[grenze+1]-grenzen[grenze])/2+grenzen[grenze], 10, str(grenze+1),horizontalalignment='center')
        #plt.savefig('GMMHistogramm_s.pdf',bbox_inches='tight')
        """

    return gmm, p

def calc_random_gaussian_mixture(gmm,gmm_n_components,n):
    #Berechnung von n Zahlen nach GMM
    ###INPUT:
    #gmm:   GaussianMixtureModel
    #gmm_n_components
    #n:     int - Anzahl der generierten Zufallszahlen
    ###OUTPUT:
    #y:     np.array / double

    gmm_n_components=gmm.weights_.shape[0]

    norm_params=np.array([])
    for i in range(0,gmm_n_components):
        norm_params=np.append(norm_params,[gmm.means_[i][0], np.sqrt(gmm.covariances_[i][0][0])])
    norm_params=np.reshape(norm_params,[gmm_n_components,2])
    
    weights = gmm.weights_
    # A stream of indices from which to choose the component
    mixture_idx = np.random.choice(len(weights), size=n, replace=True, p=weights)
    y = np.fromiter((stats.norm.rvs(*(norm_params[i])) for i in mixture_idx), dtype=np.float64)
    
    if n==1:
        y=y[0]

    return y

def calc_random_gmm_bins(gmm,gmm_n_components,low,high):
    #Bestimmung einer GMM-Zufallszahl im Intervall [low,high]
    count = 0
    gmm_n_components=gmm.weights_.shape
    random_no = calc_random_gaussian_mixture(gmm,gmm_n_components,1)
    while random_no < low or random_no > high:
        count=count+1
        random_no = calc_random_gaussian_mixture(gmm,gmm_n_components,1)
        if count >= 100:
            #print(low, ', ', high)
            random_no = (high-low)/2+low
    return random_no

def random_in_interval(hist,low,high):
    random_no = np.random.choice(hist[1],p=hist[0])
    while random_no < low or random_no > high:
        random_no = np.random.choice(hist[1],p=hist[0])
    return random_no

def approx_energy(load,deg,deltaT=0.25,plots=False):
    #Berechnung des Regressionspolynoms fuer die Energie
    ###INPUT:
    #load: DataFrame oder Array eindimensional
    #deg: Grad des Regressionspolynoms
    #deltaT, plots
    ###OUTPUT:
    #reg_poly: Koeffizienten des Regressionspolynoms

    if isinstance(load,pd.DataFrame):
        load=load.to_numpy()
    
    Y=deltaT*np.sum(load,axis=0)
    
    n=Y.shape[0]                #Laenge der Lastdaten

    Yfitting = Y / np.mean(Y,axis=0)

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

def build_example_year(load_profile,coeff_regression_energy,plots=False):
    #Erstellung eines Beispieljahres
    ###INPUT:
    #load_profile: 96x9 Matrix mit Lastprofilen (Jahreszeit, Wochentag)
    #coeff_regression_energy: Regressionspolynom der Energie
    ###OUTPUT:
    #load_example_year: Jahres-Lastgang der Laenge 96*365=35040
    
    #Importiere Zuordnung von Wochentagen und Jahreszeiten
    example_year = pd.read_excel('bspjahr.xlsx')

    #Importiere Dictionary
    season_dict, weekday_dict = loaddata.dicts_season_weekday()
    
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

    if plots:
        plt.plot(load_example_year)

    return load_example_year

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
    coeff_regression_energy = approx_energy(df_last_oneyear.loc[:,year],deg,plots=plots)

    #Normierung
    normed_load_matrix = dp.normalize_by_week(df_last_matrix)
   
    for season in range(0,3):
        weekenergy=0.25*5*normed_load_matrix[season,0].sum()+normed_load_matrix[season,1].sum()+normed_load_matrix[season,2].sum()
        for weekday in range(0,3):
            normed_load_matrix[season,weekday]=normed_load_matrix[season,weekday]/(weekenergy/7)
            normed_load_matrix[season,weekday].plot()

    load_example_year = build_example_year(normed_load_matrix,coeff_regression_energy)

    df_example_year = pd.DataFrame(load_example_year)             #in kW
    jahresenergie   = calc_energy(df_example_year)                  #in kWh

    if plots:
        plt.figure()
        df_example_year.plot(legend=False)
        plt.title('Beispieljahr im Jahresverlauf')
        plt.ylabel('Normierte Leistung')
        plt.xlabel('Zeitschritt im Jahr')
    
    load_profiles_numpy = np.zeros([96,9])
    for season in range(0,3):
        for weekday in range(0,3):
            load_profiles_numpy[:,3*season+weekday]=normed_load_matrix[season,weekday]/(jahresenergie/1000)

    iterables=[['Winter','Übergangszeit','Sommer'],['Werktag','Samstag','Sonntag']]
    ind=pd.MultiIndex.from_product(iterables, names=['Season','Weekday'])

    load_profiles = pd.DataFrame(load_profiles_numpy, columns=ind)      #in kW

    if export:
        load_profiles.to_csv('load_profiles_' + name + '.csv')
    
    return load_profiles, coeff_regression_energy

def visualize_typedays(profil, name, export_plots = False, filename=''):
    subfig, axs = plt.subplots(3,2,figsize=(20,24))
    
    season_count=0
    for season in ['Winter','Übergangszeit','Sommer']:
        day_count=0
        for day in ['Werktag','Samstag','Sonntag']:
            axs[season_count,0].plot(np.linspace(1,96,96), profil.loc[:,season].loc[:,day].to_numpy())
            axs[season_count,0].legend(['Werktag','Samstag','Sonntag'])
            axs[season_count,0].set_ylabel('Leistung [W]')
            axs[season_count,0].set_xlabel('Zeit am Tag [h]')
            axs[season_count,0].set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))
            axs[season_count,0].set_title(season)
            axs[season_count,0].grid()
            axs[day_count,1].plot(np.linspace(1,96,96), profil.loc[:,season].loc[:,day].to_numpy())
            axs[day_count,1].legend(['Winter','Übergangszeit','Sommer'])
            axs[day_count,1].set_ylabel('Leistung [W]')
            axs[day_count,1].set_xlabel('Zeit am Tag [h]')
            axs[day_count,1].set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))
            axs[day_count,1].set_title(day)
            axs[day_count,1].grid()
            day_count=day_count+1
        season_count = season_count+1
        plt.subplots_adjust(hspace=0.3)
    if export_plots:
        plt.savefig('3x2_plot_' + filename + '.pdf',bbox_inches='tight')

def mape_simulation(arr_simulation,in_measure):
    n = arr_simulation.shape[0]
    n_times = in_measure.shape[0]
    n_it = in_measure.shape[1]

    wide_simulation = np.reshape(arr_simulation,[n_times,n_it],order='F')
    #long_measure = np.reshape(df_measure.to_numpy(),[n_it,n_times])

    #long_diff=np.abs(long_measure-long_simulation)/long_measure
    if isinstance(in_measure,pd.DataFrame):
        arr_measure = in_measure.to_numpy()
    mat_diff = np.zeros(n_it)

    print(arr_measure.shape)
    print(wide_simulation.shape)

    #plt.figure()
    #plt.plot(wide_simulation)
    #plt.figure()
    #plt.plot(arr_measure)
    mat_max_power_diff = np.zeros(n_it)
    mat_energy_diff = np.zeros(n_it)

    for it in range(0,n_it):
        mat_diff[it] = 1/n_times*np.sum(np.abs(arr_measure[:,it]-wide_simulation[:,it])/arr_measure[:,it])*100

    mat_max_power_diff = np.abs(np.max(arr_measure,axis=0)-np.max(wide_simulation,axis=0))/np.max(arr_measure,axis=0)*100
    mat_energy_diff = np.abs(0.25*np.sum(arr_measure,axis=0)-0.25*np.sum(wide_simulation,axis=0))/(0.25*np.sum(arr_measure,axis=0))*100

    mat_max_power_diff_vz = (np.max(arr_measure,axis=0)-np.max(wide_simulation,axis=0))/np.max(arr_measure,axis=0)*100
    mat_energy_diff_vz = (0.25*np.sum(arr_measure,axis=0)-0.25*np.sum(wide_simulation,axis=0))/(0.25*np.sum(arr_measure,axis=0))*100

    return mat_diff, mat_max_power_diff, mat_energy_diff, mat_max_power_diff_vz, mat_energy_diff_vz