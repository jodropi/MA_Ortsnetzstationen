import dataprocessing as dp
import pandas as pd

def load_stations():
    stationen = pd.read_csv('Stationen.csv',delimiter=';')
    jahreszeiten=['Winter','Sommer','Uebergang']
    tab_seasons=pd.DataFrame({'Monat': range(1,13), 'Jahreszeit': [0,0,0,1,1,2,2,2,1,1,0,0]})
    zeiten=pd.DataFrame(pd.date_range(start='1/1/2018', periods=96, freq='15T').time)

    stationen['Cluster']=['W4','W1','W3','S','S','W2','W2','S','W1']
    stationen['Kurzname']=['A','B','C','D','E','F','G','H','I']
    stationen['Zaehler']=[99, 72, 40, 394, 382, 50, 34, 315, 90]
    group_W1=['Schule Durchholz','Guennemannshof']
    group_W2=['Querweg','Rehnocken 1']
    group_W3=['Kaemperfeld']
    group_W4=['Durchholz']
    group_S=['Lutherplatz','Marienhospital alt','Rudolf-Koenig-Strasse']

    stationen['LeistungproHA']=stationen['Leistung']/stationen['Hausanschluesse']
    stationen['LeistungproZaehler']=stationen['Leistung']/stationen['Zaehler']
    stationen['ZaehlerproHA']=stationen['Zaehler']/stationen['Hausanschluesse']
    return stationen

def load_data_from_measurements(ind_of_import: list, d=False, yd=False, ysd=False, ymd=False, ywd=False,yswd=False):
    stationen=load_stations()

    list_power_d = list()
    list_power_only_d = list()
    list_mat_power_d = list()
    list_power_yd = list()
    list_power_ysd = list()
    list_power_ymd = list()
    list_power_ywd = list()
    list_power_yswd = list()


    for i in ind_of_import:
        mat_powers={}
        filename='Lastgang Strom Station ' + stationen.loc[:,'Name'].iloc[i] + '.xlsx'
        print('Lade Daten von Station ', stationen.loc[:,'Name'].iloc[i])

        df = pd.read_excel(filename)
        df = dp.interpolate_data(df)

        if d:
            power_data_pivot_only_day = dp.pivot_data(df,bool_year=False,bool_season=False,bool_month=False,bool_weekday=False,anonomize_date=False)
            power_data_pivot = dp.pivot_data(df,bool_year=False,bool_season=True,bool_month=False,bool_weekday=True,anonomize_date=False)
            list_power_d.insert(len(list_power_d),power_data_pivot)
            list_power_only_d.insert(len(list_power_only_d),power_data_pivot_only_day)
            for season in range(0,3):
                for day in range(0,3):
                    mat_powers[season,day]=power_data_pivot.loc[:,season].loc[:,day]
            list_mat_power_d.insert(len(list_mat_power_d),mat_powers)
        if yd:
            power_data_pivot_year = dp.pivot_data(df,bool_year=True,bool_season=False,bool_month=False,bool_weekday=False,anonomize_date=False)
            list_power_yd.insert(len(list_power_yd),power_data_pivot_year)
        if ysd:
            power_data_pivot_year_season = dp.pivot_data(df,bool_year=True,bool_season=True,bool_month=False,bool_weekday=True,anonomize_date=False)
            list_power_ysd.insert(len(list_power_ysd),power_data_pivot_year_season)
        if ywd:
            power_data_pivot_year_weekday_day = dp.pivot_data(df,bool_year=True,bool_season=False,bool_month=False,bool_weekday=True,anonomize_date=False)
            list_power_ywd.insert(len(list_power_ywd),power_data_pivot_year_weekday_day)
        if yswd:
            power_data_pivot_year_season_weekday_day = dp.pivot_data(df,bool_year=True,bool_season=True,bool_month=False,bool_weekday=True,anonomize_date=False)
            list_power_yswd.insert(len(list_power_yswd),power_data_pivot_year_season_weekday_day)


    
    return list_power_d, list_power_yd, list_power_ysd, list_power_ymd, list_mat_power_d, list_power_ywd, list_power_yswd, list_power_only_d

def dicts_season_weekday():
    dict_season = {
        0: 'Winter',
        1: 'Übergangszeit',
        2: 'Sommer'
    }
    dict_weekday = {
        0: 'Werktag',
        1: 'Samstag',
        2: 'Sonntag'
    }
    return dict_season, dict_weekday

def dict_station():
    dictionarystations = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D',
        4: 'E',
        5: 'F',
        6: 'G',
        7: 'H',
        8: 'I'
    }
    return dictionarystations