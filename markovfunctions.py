import numpy as np
import pandas as pd
import dataprocessing as dp
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

def build_all_transitionmatrices(data, bins, n_steps, n_states, order=1, bool_uniform=False):
    #Bestimmen von allen Uebergangsmatrizen fuer alle Typtage
    ###INPUT:
    #data:  pd.DataFrame - Messdaten als wide-Matrix
    #bins:  matrix - Zustandsgrenzen
    #n_steps:   int - Anzahl von Zeitschritten
    #n_states:  int - Anzahl von Zustaenden
    #order:     int - Ordnung der Markov-Kette
    #bool_uniform:  boolean - keine Jahreszeiten?
    ###OUTPUT:
    #all_transitionmatrices:    Matrix[3,3] -> np.array([n_states,n_states]) - Alle Uebergangsmatrizen
    #all_bins:                  Matrix[3,3] -> np.array([n_states]) - Alle Zustandsgrenzen

    print('>>> Berechne alle Uebergangsmatrizen...')
    all_transitionmatrices = {}
    all_bins = {}

    if bool_uniform:
        a_transitionmatrix, a_bin = build_transitionmatrix(data=data,bins=bins,n_steps=n_steps,order=order,n_states=n_states)
        for season in range(0,3):
            for weekday in range(0,3):
                #print('Jahreszeit =', season, ', Wochentag =', weekday)
                all_bins[season,weekday] = a_bin
                all_transitionmatrices[season,weekday] = a_transitionmatrix
    else:
        for season in range(0,3):
            for weekday in range(0,3):
                data_part = data.loc[:,season].loc[:,weekday]
                #print('Jahreszeit =', season, ', Wochentag =', weekday)
                all_transitionmatrices[season,weekday], all_bins[season,weekday] = build_transitionmatrix(data=data_part,bins=bins,n_steps=n_steps,order=order,n_states=n_states)

    print('>>> Berechnung der Uebergangsmatrizen abgeschlossen!')
    return all_transitionmatrices, all_bins

def build_transitionmatrix(data, bins, n_steps, n_states, order=1,proof_tm=False):
    #Bestimmen von einer Uebergangsmatrix
    ###INPUT:
    #data:  pd.DataFrame - Messdaten als wide-Matrix
    #bins:  np.array[n_states+1] - Zustandsgrenzen
    #n_steps:   int - Anzahl von Zeitschritten
    #n_states:  int - Anzahl von Zustaenden
    #order:     int - Ordnung der Markov-Kette
    ###OUTPUT:
    #transitionmatrix_list: list -
    #bins_list:             list -

    print('>>> Berechne Uebergangsmatrix...')

    if len(bins) == 0:
        dynamic_bins = True             #Falls keine Grenzen hinterlegt sind, dynamische Grenzen bilden
    else:
        dynamic_bins = False

    n_data=data.shape[1]

    transitionmatrix_list = list()
    bins_list = list()

    if order == 1:
        #Zustandsgrenzen
        if dynamic_bins:
            for step in range(0,n_steps):
                #Fur jeden Zeitschritt Zustandsgrenzen bilden
                _, this_bins = pd.cut(data.iloc[step,:].to_numpy().reshape(-1), n_states,retbins=True)
                bins_list.insert(len(bins_list), this_bins)
        else:
            for step in range(0,n_steps):
                bins_list.insert(len(bins_list), bins)

        #Hauefigkeiten zaehlen
        for step in range(0,n_steps):
            countmatrix = np.zeros([n_states, n_states])
            transitionmatrix = np.zeros([n_states, n_states])

            data_start_state = data.iloc[step,:]
            data_end_state = data.iloc[(step+1)%96,:]
            
            bins_start = bins_list[step]
            bins_end = bins_list[(step+1)%96]

            #Hauefigkeiten zaehlen
            for i in range(0,n_data):
                for start_state in range(0,n_states):
                    if data_start_state.iloc[i] >= bins_start[start_state] and data_start_state.iloc[i] <= bins_start[start_state+1]:
                        for end_state in range(0,n_states):
                            if data_end_state.iloc[i] >= bins_end[end_state] and data_end_state.iloc[i] <= bins_end[end_state+1]:
                                #print('Gefunden')
                                countmatrix[start_state,end_state]=countmatrix[start_state,end_state]+1
            
            sum_columns=np.sum(countmatrix,axis=1)

            #Uebergangswahrscheinlichkeiten zaehlen
            for start_state in range(0,n_states):
                for end_state in range(0,n_states):
                    if sum_columns[start_state] != 0:
                        transitionmatrix[start_state,end_state]=countmatrix[start_state,end_state]/sum_columns[start_state]

            transitionmatrix_list.insert(len(transitionmatrix_list), transitionmatrix)
        
        if proof_tm:
            transitionmatrix_list=proof_transistionmatrix(transitionmatrix_list,n_steps,n_states)
    
    elif order == 2:
        for step in range(0,n_steps):
            countmatrix = np.zeros([n_states**order, n_states])
            transitionmatrix = np.zeros([n_states**order, n_states])
            data_before_state = data.iloc[(step-1)%96,:]
            data_start_state = data.iloc[step,:]
            
            data_end_state = data.iloc[(step+1)%96,:]
            
            for i in range(0,n_data):
                print('Before =', data_before_state.iloc[i], 'Start = ', data_start_state.iloc[i], ', Ende = ', data_end_state.iloc[i])
                for before_state in range(0,n_states):
                    for start_state in range(0,n_states):
                        #print(i, n_data)
                        if data_before_state.iloc[i] >= bins[before_state] and data_before_state.iloc[i] <= bins[before_state+1]:
                            if data_start_state.iloc[i] >= bins[start_state] and data_start_state.iloc[i] <= bins[start_state+1]:
                                for end_state in range(0,n_states):
                                    if data_end_state.iloc[i] >= bins[end_state] and data_end_state.iloc[i] <= bins[end_state+1]:
                                        #print('Gefunden')
                                        countmatrix[before_state*n_states+start_state,end_state]=countmatrix[before_state*n_states+start_state,end_state]+1
            
            sum_columns=np.sum(countmatrix,axis=1)
            for state_combination in range(0,n_states**order):
                for end_state in range(0,n_states):
                    if sum_columns[state_combination] != 0:
                        transitionmatrix[state_combination,end_state]=countmatrix[state_combination,end_state]/sum_columns[state_combination]

            transitionmatrix_list.insert(len(transitionmatrix_list), transitionmatrix)
    
    return transitionmatrix_list, bins_list

def proof_transistionmatrix(tm_list, n_steps, n_states):
    for timestep in range(1,n_steps+1):
        actual_tm = tm_list[timestep%n_steps]
        previous_tm = tm_list[(timestep-1)%n_steps]

        summe_actual_tm = np.sum(actual_tm,axis=1)
        summe_previous_tm = np.sum(previous_tm,axis=0)


        for i in range(0,n_states):
            if summe_actual_tm[i] == 0:
                if summe_previous_tm[i] > 0:
                    for j in range(0,n_states):
                        if previous_tm[j,i] > 0:
                            #print('Zeit', timestep, ':', actual_tm[i,:])
                            #print('Problem in Zeitschritt', timestep, ' in State', i, 'Ursprung', j, ' mit Ws', previous_tm[j,i])
                            tm_list[timestep-1][j,i]=0
                            #print(tm_list[timestep-1][j,i])

        for state_start in range(0,n_states):                                     #Normierung
            summe=np.sum(tm_list[timestep-1],axis=1)
            if summe[state_start] != 0:
                tm_list[timestep-1][state_start,:]=tm_list[timestep-1][state_start,:]/summe[state_start]

    return tm_list

def create_bins(data, case_standardisation, n_states):
    #Normierungsoptionen aufschluesseln
    case_1 = case_standardisation[0]
    case_2 = case_standardisation[1]
    case_3 = case_standardisation[2]
    case_4 = case_standardisation[3]

    #Grenzen erstellen
    if case_3 == 0:
        if case_1 == 2:
            bins = np.linspace(0,1,n_states+1)
        else:
            _, bins = pd.cut(data.to_numpy().reshape(-1), n_states,retbins=True)
    else:
        bins = []
    
    return bins

def build_randomwalk(transitionmatrix_list,bins,n_states,n_steps,last_values_before,weekday,season,gmm,gmm_n_components,order=1,dynamic_gmm=True):
    #Bestimmung eines zufaelligen Weges
    ###INPUT:
    # transitionmatrix_list:    list - Uebergangsmatrizen
    # bins:                     np.array - Zustandsgrenzen
    # n_states                  int
    # n_steps                   int
    # last_values_before:       int - Leistung zu t=-1
    # weekday                   int - 0 Werktag, 1 Samstag, 2 Sonntag
    # season                    int - 0 Winter, 1 Uebergang, 2 Sommer
    # gmm                       GaussianMixtureModel / list -> GaussianMixtureModel
    # gmm_n_components          
    # order                     int
    # dynamic_gmm
    ###OUTPUT:
    # random_power:             np.array - Zufaelliger Weg
    #                           np.array / int - Letzte/r Wert/e

    random_walk=np.zeros([n_steps],int)
    random_power=np.zeros([n_steps])
    
    if order == 1:    #Markov-Kette erster Ordnung                                                          
        print('Starte Random-Walk an Tag', weekday, ' in Jahreszeit', season, ' mit Voergaenger-Leistung', last_values_before)

        ###Start-Zustand bestimmen
        sum_start_states = np.sum(transitionmatrix_list[n_steps-1],axis=1)

        ###Welche Start-States sind verfuegbar?
        start_state_indizes=np.zeros(0, int)
        for i in range(0,n_states):
            if sum_start_states[i]>0:
                start_state_indizes=np.append(start_state_indizes, [i])
        this_bins=bins[n_steps-1]
        ###In welchem Start-State ist der initial_value?
        i=0
        while i <= n_states and last_values_before[0]>=this_bins[i]:
            i=i+1
        before_state=i-1

        if before_state in start_state_indizes:
            before_state=before_state
        else:
            distance_to_possible_states = np.abs(start_state_indizes-before_state)
            before_state = start_state_indizes[np.argmin(distance_to_possible_states)]
            print('Vorgaenger-Leistung nicht zulaessig! Neuer Vorgaenger-Zustand:', before_state)
        
        transitionmatrix=transitionmatrix_list[n_steps-1]
        ind = before_state
        random_walk[0] = np.random.choice(n_states,1,p=transitionmatrix[ind])

        ##################################################################
        for step in range(0,n_steps-1):
            transitionmatrix=transitionmatrix_list[step]
            random_walk[step+1]=np.random.choice(n_states,1,p=transitionmatrix[random_walk[step]])
        
        for i in range(0,n_steps):
            this_bins = bins[i]
            if dynamic_gmm:
                this_gmm=gmm[i]
            else:
                this_gmm=gmm
            random_power[i]=calc_random_gmm_bins(this_gmm,gmm_n_components,this_bins[random_walk[i]],this_bins[random_walk[i]+1])   

        return random_power, [random_power[n_steps-1]]
    elif order == 2:
        random_walk=np.zeros([n_steps],int)
        random_power=np.zeros([n_steps])
        trans=np.zeros([1,n_states])

        last_states_before = np.zeros(order)
        
        ###In welchem Start-State ist der initial_value?
        for j in range(0,order):
            i=0
            while i <= n_states and last_values_before[j] >= bins[i]:
                i=i+1
            last_states_before[j]=i-1

        #Welche Vorgaenger-States sind zulaessig?
        sum_start_states = np.sum(transitionmatrix_list[n_steps-1],axis=1)
        start_before_indizes_1=np.zeros(0, int)
        start_before_indizes_2=np.zeros(0, int)
        for i in range(0,n_states**order):
            if sum_start_states[i] > 0:
                trans=np.append(trans,[transitionmatrix_list[95][i,:]],axis=0)
                start_before_indizes_1=np.append(start_before_indizes_1, [int(i/n_states)])
                start_before_indizes_2=np.append(start_before_indizes_2, [i%n_states])
        anz=start_before_indizes_1.shape[0]
        
        #Ist die Vorgabe zulaessig?

        if last_states_before[0] in start_before_indizes_1:
            combination = np.zeros(0,int)
            for i in range(0,anz):
                if start_before_indizes_1[i]==last_states_before[0]:
                    combination = np.append(combination,start_before_indizes_2[i])
            if last_states_before[1] in combination:
                last_states = last_states_before
                print('Vorgaenger-Leistung unveraendert')
            else:
                last_states = [last_states_before[0], np.random.choice(combination,1)[0]]
                print('Vorgaenger-Leistung -1 veraendert')
        else:
            if last_states_before[1] in start_before_indizes_2:
                combination = np.zeros(0,int)
                for i in range(0,anz):
                    if start_before_indizes_2[i]==last_states_before[1]:
                        combination = np.append(combination,start_before_indizes_1[i])
                
                last_states = [np.random.choice(combination,1)[0],last_states_before[1]]
                print('Vorgaenger-Leistung -2 veraendert')
            else:
                ind=np.random.randint(0,anz)
                print(ind)
                last_states = [start_before_indizes_1[ind], start_before_indizes_2[ind]]
                print('Vorgaenger-Leistungen veraendert:', last_states)

        transitionmatrix=transitionmatrix_list[95]
        ind=int(last_states[0]*n_states+last_states[1])
        if np.sum(transitionmatrix[ind])<= 0.95:
            print(weekday, season, ',', last_states, ',', transitionmatrix[ind])
            print(start_before_indizes_1, start_before_indizes_2)
        random_walk[0] = np.random.choice(n_states,1,p=transitionmatrix[ind])
        transitionmatrix=transitionmatrix_list[0]
        ind=int(last_states[1]*n_states+random_walk[0])
        random_walk[1] = np.random.choice(n_states,1,p=transitionmatrix[ind])

        for step in range(1,n_steps-1):
            transitionmatrix=transitionmatrix_list[step]
            random_walk[step+1]=np.random.choice(n_states,1,p=transitionmatrix[random_walk[(step-1)%n_steps]*n_states+random_walk[step]])
            
        for i in range(0,n_steps):
            random_power[i]=calc_random_gmm_bins(gmm,gmm_n_components,bins[random_walk[i]],bins[random_walk[i]+1])  
        return random_power, random_power[n_steps-2:n_steps]
    else:
        return random_power, [0]