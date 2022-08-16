from distutils.log import error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import loaddata
import math
from scipy.stats import norm
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from datetime import datetime, timedelta
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score

def distance(x,y):
    return np.linalg.norm(x-y)

def distance_matrix(partition):
    N=np.shape(partition)[1]
    mat=np.zeros([N,N])
    for i in range(0,N):
        for j in range(0,N):
            mat[i,j]=distance(partition[:,i],partition[:,j])
    return mat

def CDI(centroids,df,label):
    n_clusters=np.shape(centroids)[1]
    sum_clusters = 0
    for i in range(0,n_clusters):
        ind=np.zeros(0,int)
        for j in range(0,9):
            if label[j]==i:
                ind=np.append(ind,j)
        sum_clusters = sum_clusters+infrasetdist(df[:,ind])**2
    D=infrasetdist(centroids)
    
    if D == 0:
        print('Problem')
    
    return 1/infrasetdist(centroids)*np.sqrt(1/n_clusters*sum_clusters)
    
def infrasetdist(set_time_series):
    N=np.shape(set_time_series)[1]
    distance_mat=distance_matrix(set_time_series)
    hat_d=np.sqrt(1/(2*N)*np.sum(distance_mat**2))
    return hat_d

def build_winter_summer_week_profile(list_mat_power_d, n_steps = 96):
    n_stations = len(list_mat_power_d)

    df_to_cluster=np.zeros([n_stations,6*n_steps])
    for i in range(0,n_stations):
        A=list_mat_power_d[i][0,0].mean(axis=1).to_numpy()
        B=list_mat_power_d[i][0,1].mean(axis=1).to_numpy()
        C=list_mat_power_d[i][0,2].mean(axis=1).to_numpy()
        D=np.append(np.append(A,B),C)
        A=list_mat_power_d[i][2,0].mean(axis=1).to_numpy()
        B=list_mat_power_d[i][2,1].mean(axis=1).to_numpy()
        C=list_mat_power_d[i][2,2].mean(axis=1).to_numpy()
        E=np.append(np.append(A,B),C)

        df_to_cluster[i,:]=np.append(D,E)
        df_to_cluster[i,:]=(df_to_cluster[i,:]-np.min(df_to_cluster[i,:]))/(np.max(df_to_cluster[i,:])-np.min(df_to_cluster[i,:]))              #Min-Max-Normalisierung
    

    return df_to_cluster

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
    # set_link_color_palette(['g', 'r', 'c', 'm', 'y', 'k'])
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix,**kwargs)

def analyze_cluster_indicators(df, plot_Dendrogramm = True, plot_CDI = True, plot_DBI = False, export_plots=False):

    DBI_mat = np.zeros(8)
    CDI_mat = np.zeros(8)
    list_models=list()
    for n in range(2,9):
        X=df
        Y=X.T
        
        mymodel_n = AgglomerativeClustering(n_clusters=n,compute_distances=True)
        mymodel_n = mymodel_n.fit(X)

        centroids = np.zeros([2*288,n])
        label=mymodel_n.labels_
        #differenz=np.diff(mymodel_n.distances_)
        #plt.figure()
        #plt.plot(np.arange(9,1,-1), mymodel_n.distances_)
        #print(mymodel_n.distances_)
        #plt.figure()
        
        for i in range(0,n):
            ind=np.zeros(0,int)
            for j in range(0,9):
                if label[j]==i:
                    ind=np.append(ind,j)
            centroids[:,i]=np.mean(Y[:,ind],axis=1)

        list_models.insert(len(list_models),mymodel_n)

        DBI_mat[n-2] = davies_bouldin_score(X, label)
        CDI_mat[n-2] = CDI(centroids,Y,label)
    
    if plot_Dendrogramm:
        plot_dendrogram(list_models[0],truncate_mode="level", p=5, labels=['A','B','C','D','E','F','G','H','I'])
        if export_plots:
            plt.savefig('export/Dendrogramm_Alle.pdf',bbox_inches='tight')
    """
    subfig, axs = plt.subplots(1,2,figsize=(12.8,4.8))
    
    no=0
    axs[no].plot(np.linspace(2,9,8),DBI_mat)
    axs[no].set_xlabel('Anzahl von Clustern')
    axs[no].set_ylabel('DBI')

    no=1
    axs[no].plot(np.linspace(2,9,8),CDI_mat)
    axs[no].set_xlabel('Anzahl von Clustern')
    axs[no].set_ylabel('CDI')
    """
    if plot_CDI:
        plt.figure()
        plt.plot(np.linspace(2,9,8),CDI_mat)
        plt.xlabel('Anzahl von Clustern')
        plt.ylabel('CDI')

    if plot_DBI:
        plt.figure()
        plt.plot(np.linspace(2,9,8),DBI_mat)
        plt.xlabel('Anzahl von Clustern')
        plt.ylabel('DBI')

def plot_week_profiles_cluster(df,cluster_label):
    dictionarystations=loaddata.dict_station()
    n_clusters = int(cluster_label.max()+1)
    n_stations = 9

    cluster_no=['2', '5', '1', '3', '4']

    for i in range(0,n_clusters):
        ind=np.zeros(0,int)
        for j in range(0,n_stations):
            if cluster_label[j]==i:
                ind=np.append(ind,j)
        if len(ind) == 1:
            label_ind = dictionarystations[ind[0]]
        else:
            label_ind = list()
            for j in range(0,len(ind)):
                label_ind.insert(len(label_ind),dictionarystations[ind[j]])
        
        plt.figure(figsize=(19.2,4.8))
        plt.plot(df[ind,:].T,label=label_ind)
        plt.xticks([48,144,240,336,432,528],['Winter Werktag', 'Winter Samstag','Winter Sonntag','Sommer Werktag', 'Sommer Samstag', 'Sommer Sonntag'])
        plt.axvline(96,color='black')
        plt.axvline(192,color='black')
        plt.axvline(288,color='black')
        plt.axvline(384,color='black')
        plt.axvline(480,color='black')
        plt.ylabel('Min-Max-normierte Leistung')
        plt.xlim([0,576])
        plt.grid()
        plt.title('Cluster ' + cluster_no[i])
        leg = plt.legend()
        plt.savefig('export/Wochenlastgang_Cluster_' + cluster_no[i] + '.pdf', bbox_inches = 'tight')