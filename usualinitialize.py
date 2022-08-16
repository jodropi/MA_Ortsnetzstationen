import numpy as np
import pandas as pd
import dataprocessing as dp
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def dowhatineed():
    stationen = pd.read_csv('Stationen.csv',delimiter=';')
    jahreszeiten=['Winter','Sommer','Uebergang']
    tab_seasons=pd.DataFrame({'Monat': range(1,13), 'Jahreszeit': [0,0,0,1,1,2,2,2,1,1,0,0]})
    zeiten=pd.DataFrame(pd.date_range(start='1/1/2018', periods=96, freq='15T').time)

    stationen['Cluster']=['W4','W1','W3','S','S','W2','W2','S','W1']
    stationen['Zaehler']=[99, 72, 40, 394, 382, 50, 34, 315, 90]
    group_W1=['Schule Durchholz','Guennemannshof']
    group_W2=['Querweg','Rehnocken 1']
    group_W3=['Kaemperfeld']
    group_W4=['Durchholz']
    group_S=['Lutherplatz','Marienhospital alt','Rudolf-Koenig-Strasse']

    stationen['LeistungproHA']=stationen['Leistung']/stationen['Hausanschluesse']
    stationen['LeistungproZaehler']=stationen['Leistung']/stationen['Zaehler']
    stationen['ZaehlerproHA']=stationen['Zaehler']/stationen['Hausanschluesse']

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": '16'
    })
    return stationen