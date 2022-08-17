###IMPORT
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from dataprocessing import norm_data_for_simulation, MPE
from markovfunctions import build_all_transitionmatrices, proof_transistionmatrix, create_bins, build_randomwalk
from loadmodelingfunctions import calc_energy, calc_regression_polynoms, calc_random_gaussian_mixture, calc_random_gmm_bins, calc_gaussian_mixture, random_in_interval, calc_daily_energy
import matplotlib.pyplot as plt

directory_export = 'export/'

#####################################################################################################
class StochasticLoadModel:
    def __init__(
        self,
        n_states = 10,
        n_steps = 96,
        n_days = 365,
        n_comp_gmm = 4,
        order_regression = 4,
        order_markov=1,
        normalization = [0,0,0,1],
        bool_no_seasons=False,
        bool_dynamic_gmm=True
    ):
        self.n_states = n_states
        self.n_steps = n_steps
        self.n_days = n_days
        self.n_comp_gmm = n_comp_gmm
        self.order_regression = order_regression
        self.order_markov=order_markov
        self.normalization = normalization
        self.bool_no_seasons=bool_no_seasons
        self.bool_dynamic_gmm=bool_dynamic_gmm

        self.transitionmatrices = None
        self.bins = None
        self.gmm = None
        self.rescale_parameters = None
    
    def parameterize(self, data_in, plots = False):
        print('>>> Parmetrisiere ein stochastisches Modell...')
        data          = data_in[0]                              #DataFrame mit Leveln Jahreszeiten, Wochentagen, Datum
        data_one_year = data_in[1]                              #DataFrame mit Leveln Jahren, Datum
        data_one_year_weekday = data_in[2]                      #DataFrame mit Leveln Jahren, Datum

        data_for_simulation, extreme_values_data = norm_data_for_simulation(data, self.normalization)

        energy_per_year, c_factors, extreme_vec, std_factors, std_vec, true_values, factors_weekdays, std_weekdays = calc_regression_polynoms(data_in=data_one_year,data_weekday_in=data_one_year_weekday,deg=self.order_regression,case_normierung=self.normalization,plots=plots)

        self.rescale_parameters=[energy_per_year, c_factors, extreme_vec[1], std_factors, extreme_vec[2], extreme_vec[0], extreme_values_data, true_values, factors_weekdays, std_weekdays]

        if self.bool_dynamic_gmm:
            gmm = list()
            for timestep in range(0,96):
                #if timestep == 40:
                #    new_gmm, _ = calc_gaussian_mixture(data_for_simulation.iloc[timestep].to_numpy(),4,plots=True)
                #else:
                new_gmm, _ = calc_gaussian_mixture(data_for_simulation.iloc[timestep].to_numpy(),4)
                gmm.insert(len(gmm),new_gmm)
        else:
            gmm , _ = calc_gaussian_mixture(data_for_simulation.to_numpy(), self.n_comp_gmm)
        self.gmm = gmm

        bins = create_bins(data_for_simulation, self.normalization, self.n_states)

        self.transitionmatrices, self.bins = build_all_transitionmatrices(data_for_simulation, bins, bool_uniform=self.bool_no_seasons, n_steps=self.n_steps, n_states=self.n_states, order=self.order_markov)

        print('>>> Parametrisierung abgeschlossen!')

    def simulate(self, n_runs=1, daytypes_by_file='bspjahr.xlsx'):
        # Simulation durch eine Markov-Kette
        ###INPUT:
        # tm_matrix:        Matrix
        # bins_matrix:      Matrix
        # n_states
        # n_steps
        # n_days
        # n_runs
        # gmm
        # gmm_n_components
        ###OUTPUT
        # results_power:    np.array mit shape [n_steps, n_runs]

        print('>>> SIMULATION GESTARTET...')
        
        example_year_meta = pd.read_excel('resources/' + daytypes_by_file)
        results_power = np.zeros([self.n_steps*self.n_days,n_runs])
        before_values = np.zeros(self.order_markov)

        for it in range(0,n_runs):
            print('>>> Starte Markov-Ketten-Simulation - Iteration ' + str(it))
            simulated_power = np.zeros(self.n_steps*self.n_days)
            
            for day in range(0,self.n_days):
                print('Tag', day)
                i_begin = day*self.n_steps
                weekday=example_year_meta['Tag'].iloc[day]
                season=example_year_meta['Jahreszeit'].iloc[day]
                tm_day=self.transitionmatrices[season,weekday]
                bins_day=self.bins[season,weekday]
                simulated_power[i_begin:i_begin+self.n_steps], before_values = build_randomwalk(tm_day, bins_day, self.n_states, self.n_steps, before_values, weekday, season, self.gmm, 4)
            results_power[:, it] = simulated_power

        if self.normalization[0] == 0:
            bool_normed = False
        else:
            bool_normed = True

        result = StochasticSimulationResult(self, results_power, self.bins, n_runs, normed=bool_normed)

        return result
    
    def checkconvergence(self, typedays, n_it = 100):
        X=np.arange(0,n_it)
        plt.figure(figsize=(20,10))
        for season in range(0,3):
            for weekday in range(0,3):
                results_power = np.zeros([self.n_steps,n_it])
                tm_day=self.transitionmatrices[season,weekday]
                bins_day=self.bins[season,weekday]
                for it in range(0,n_it):
                    before_values = np.array([np.random.uniform(0, 1)])
                    results_power[:,it] , _ = build_randomwalk(tm_day, bins_day, self.n_states, self.n_steps, before_values, weekday, season, self.gmm, 4)
                typeday_power = typedays[season,weekday]
                RMSE_mat = np.zeros(n_it)
                for it in range(1,n_it+1):
                    RMSE_mat[it-1] = np.sqrt(1/self.n_steps*np.sum((np.mean(results_power[:,0:it],axis=1)-typeday_power)**2))
                plt.plot(X,RMSE_mat,label='season = ' + str(season) + ', weekday = ' + str(weekday) )
        leg = plt.legend()

    def denormalize(self, result, energy=0, new_case_4=-1, max_value=0):
        simulation_matrix = np.copy(result.result_matrix)

        case_1 = self.normalization[0]
        case_2 = self.normalization[1]
        case_3 = self.normalization[2]
        if new_case_4==-1:
            case_4 = self.normalization[3]
        else:
            case_4 = new_case_4

        n_runs = result.n_runs
        n_steps_total = self.n_days*self.n_steps

        #Erste Stelle: 0: Keine Normierung, 1: Normierung auf Mittelwert, 2: Normierung auf Min/Max, 3: z-Standardisierung
        #Zweite Stelle: 0: Lokale Normierung, 1: Globale Normierung
        #Dritte Stelle: 0: Feste Grenzen, 1: Dynamische Grenzen
        #Vierte Stelle: 0: Faktor, 1: Absolut, 2: Wahre Werte

        energy_per_year     = self.rescale_parameters[0]
        extreme_values_data = self.rescale_parameters[6]

        c_min               = self.rescale_parameters[1][0]
        c_max               = self.rescale_parameters[1][2]
        mean_vec            = self.rescale_parameters[2]
        max_vec             = self.rescale_parameters[4]
        min_vec             = self.rescale_parameters[5]
        
        std_min             = self.rescale_parameters[3][0]
        std_max             = self.rescale_parameters[3][2]

        true_min_vec        = self.rescale_parameters[7][0]
        true_mean_vec       = self.rescale_parameters[7][1]
        true_max_vec        = self.rescale_parameters[7][2]

        factors_weekday     = self.rescale_parameters[8]
        std_weekday     = self.rescale_parameters[9]

        #factors_weekdays_0  = self.rescale_parameters[8][0]
        #factors_weekdays_1  = self.rescale_parameters[8][1]
        #factors_weekdays_2  = self.rescale_parameters[8][2]
        
        example_year = pd.read_excel('resources/bspjahr.xlsx')

        #print('Normierung:', case_1, ', Lokal/Global:', case_2, ', Grenzen:', case_3)
        extended_matrix=np.zeros([self.n_days*self.n_steps,n_runs])
        p_bar = energy_per_year/(self.n_days*24)

        print('Gewuenschte Jahresenergie: ', str(energy_per_year), ', resultierende mittlere Jahresleistung:', str(p_bar))

        scaled_wide_matrix = np.zeros([self.n_days,self.n_steps])
        scaled_bins=np.zeros([self.n_days,self.n_states+1])

        if case_1 == 0:
            extended_matrix=simulation_matrix
        elif case_1 == 1:
            if case_2 == 0:
                for it in range(0,n_runs):
                    wide_matrix=np.reshape(simulation_matrix[:,it],[self.n_days,self.n_steps])
                    for day in range(0,self.n_days):
                        if case_4 == 0:
                            scaled_wide_matrix[day,:]=mean_vec[day]*p_bar*wide_matrix[day,:]
                        elif case_4 == 1:
                            scaled_wide_matrix[day,:]=mean_vec[day]*wide_matrix[day,:]
                    extended_matrix[:,it]=np.reshape(scaled_wide_matrix,n_steps_total)
            else:
                for it in range(0,n_runs):
                    extended_matrix = extreme_values_data[1]*simulation_matrix
        elif case_1 == 2:
            if case_2 == 0:
                for it in range(0,n_runs):
                    wide_matrix=np.reshape(simulation_matrix[:,it],[self.n_days,self.n_steps])
                    for day in range(0,self.n_days):
                        if case_4 == 0:
                            scaled_wide_matrix[day,:]=mean_vec[day]*p_bar*(wide_matrix[day,:]*(c_max-c_min)+c_min)
                        elif case_4 == 1 or case_4 == 3:
                            scaled_wide_matrix[day,:]=(wide_matrix[day,:]*(max_vec.mean()-min_vec.mean())+min_vec.mean())
                        elif case_4 == 2:
                            scaled_wide_matrix[day,:]=(wide_matrix[day,:]*(true_max_vec[day]-true_min_vec[day])+true_min_vec[day])
                    extended_matrix[:,it]=np.reshape(scaled_wide_matrix,n_steps_total)
            else:
                extended_matrix = simulation_matrix*(extreme_values_data[2]-extreme_values_data[0])+extreme_values_data[0]
        elif case_1 == 4:
            if case_2 == 0 or case_2==2:
                for it in range(0,n_runs):
                    wide_matrix=np.reshape(simulation_matrix[:,it],[self.n_days,self.n_steps])
                    for day in range(0,self.n_days):
                        if case_4 == 0:
                            scaled_wide_matrix[day,:]=max_vec[day]*p_bar*wide_matrix[day,:]
                        elif case_4 == 1:
                            scaled_wide_matrix[day,:]=max_vec[day]*wide_matrix[day,:]
                        elif case_4 == 2:
                            scaled_wide_matrix[day,:]=true_max_vec[day]*wide_matrix[day,:]
                        elif case_4 == 3:
                            weekday=example_year.iloc[day,1]
                            season=example_year.iloc[day,2]
                            factor_of_the_day = np.random.normal(factors_weekday[season,weekday],std_weekday[season,weekday])
                            scaled_wide_matrix[day,:]=factor_of_the_day*max_vec[day]*wide_matrix[day,:]
                    extended_matrix[:,it]=np.reshape(scaled_wide_matrix,n_steps_total)
            elif case_2 == 1:
                for it in range(0,n_runs):
                    if case_4 == 4:
                        extended_matrix = max_value*simulation_matrix
                    else:
                        extended_matrix = extreme_values_data[2]*simulation_matrix
        else:
            #z-Score
            extended_matrix=np.zeros([self.n_days*self.n_steps,self.n_runs])
            p_bar = energy_per_year/(self.n_days*24)
            for it in range(0,n_runs):
                wide_matrix=np.reshape(simulation_matrix[:,it],[self.n_days,self.n_steps])
                scaled_wide_matrix = np.zeros([self.n_days,self.n_steps])
                extended_matrix[:,it]=np.reshape(scaled_wide_matrix,n_steps_total)

        for it in range(0,n_runs):
            energy_of_the_iteration = calc_energy(extended_matrix[:,it])
            extended_matrix[:,it] = extended_matrix[:,it]*energy/energy_of_the_iteration

        denormalized_result = StochasticSimulationResult(self, extended_matrix, scaled_bins, n_runs, normed = False)
        
        print('>>> Simulation erfolgreich denormalisiert!')

        return denormalized_result

#####################################################################################################################

class StochasticSimulationResult:
    def __init__(
        self,
        model,
        result_matrix,
        bins,
        n_runs,
        normed=False
    ):
        self.n_states = model.n_states
        self.n_steps = model.n_steps
        self.n_days = model.n_days
        self.n_runs = n_runs
        self.normed = normed
        self.model = model
        self.result_matrix = result_matrix

    def analyze(
            self,
            reference_data_in,
            plots = True,
            random_plots = False,
            show_tables = False,
            export_plots = False,
            filename_plots = ''
        ):
        
        normed_analysis = self.normed
        simulation_matrix = self.result_matrix
        n_runs = self.n_runs
        n_states = self.n_states
        n_steps = self.n_steps
        n_days = self.n_days

        if normed_analysis:
            reference_data_wide, _ = norm_data_for_simulation(reference_data_in,self.model.normalization)
            reference_data_wide=reference_data_wide.to_numpy()
        else:
            if isinstance(reference_data_in, pd.DataFrame):
                reference_data_wide = reference_data_in.to_numpy()
            else:
                reference_data_wide = reference_data_in

        reference_data  = np.reshape(reference_data_wide,[n_steps*n_days],order='F')
        tages_messdaten = np.reshape(reference_data,[365,96])

        mean_messdaten = np.mean(tages_messdaten,axis=0)
        std_messdaten = np.std(tages_messdaten,axis=0)
        max_messdaten_pro_zeit = np.max(tages_messdaten,axis=0)
        max_messdaten = np.max(tages_messdaten,axis=1)
        max_max_messdaten = np.max(max_messdaten)
        mean_max_messdaten = np.mean(max_messdaten)
        
        simulation_mean=np.zeros([96,n_runs])
        simulation_std=np.zeros([96,n_runs])
        simulation_max_pro_Zeit=np.zeros([96,n_runs])
        simulation_max=np.zeros([365,n_runs])
        simulation_min=np.zeros([365,n_runs])
        simulation_meanstrich=np.zeros([365,n_runs])
        
        for k in range(0,n_runs):
            simulation_mean[:,k]=np.mean(np.reshape(simulation_matrix[:,k],[365,96]),axis=0)
            simulation_std[:,k]=np.std(np.reshape(simulation_matrix[:,k],[365,96]),axis=0)
            simulation_max_pro_Zeit[:,k]=np.max(np.reshape(simulation_matrix[:,k],[365,96]),axis=0)
            simulation_max[:,k]=np.max(np.reshape(simulation_matrix[:,k],[365,96]),axis=1)
            simulation_min[:,k]=np.min(np.reshape(simulation_matrix[:,k],[365,96]),axis=1)
            simulation_meanstrich[:,k]=np.mean(np.reshape(simulation_matrix[:,k],[365,96]),axis=1)
        
        max_max_simulation=np.max(simulation_max_pro_Zeit,axis=1)
        mean_mean_simulation=np.mean(simulation_mean,axis=1)
        mean_std_simulation=np.mean(simulation_std,axis=1)
        mean_max_simulation=np.mean(simulation_max,axis=0)
        mean_min_simulation=np.mean(simulation_min,axis=0)
        mean_meanstrich_simulation=np.mean(simulation_meanstrich,axis=0)
        max_max_simu=np.max(simulation_max)

        #print(mean_mean_simulation.shape)

        MAPE_mean = 1/n_steps*(np.sum(np.abs(mean_mean_simulation-mean_messdaten)/mean_messdaten))*100
        MAPE_std = 1/n_steps*(np.sum(np.abs(mean_std_simulation-std_messdaten)/std_messdaten))*100
        MAPE_max = (np.abs(max_max_simu-max_max_messdaten)/max_max_messdaten)*100

        print('MAPE von mean = ', MAPE_mean, '% , MAPE von std = ', MAPE_std, '%')
        print('MAX Sim = ', np.mean(mean_max_simulation), ', MAX Mess = ', mean_max_messdaten, ', Abs Delta =', np.abs(np.mean(mean_max_simulation)-mean_max_messdaten)/mean_max_messdaten*100)

        #simulation_mean_year=np.mean(simulation_matrix,axis=1)
        all_sim_data=list()
        for k in range(0,n_runs):
            all_sim_data.insert(len(all_sim_data),np.reshape(simulation_matrix[:,k],[365,96]))

        ind_statistics=['count','mean','std','min','25%','50%','75%','max']

        statistics_describe=pd.DataFrame(index=ind_statistics)
        statistics_describe_mean=pd.DataFrame(index=ind_statistics)
        statistics_describe_min=pd.DataFrame(index=ind_statistics)
        statistics_describe_max=pd.DataFrame(index=ind_statistics)
        statistics_describe_min_vz=pd.DataFrame(index=ind_statistics)
        statistics_describe_max_vz=pd.DataFrame(index=ind_statistics)
        statistics_energy=pd.DataFrame(index=['Jahresenergie [kWh]'])
        statistics_index=pd.DataFrame(index=['argmax','argmin'])

        if normed_analysis:
            label_leistung='Normierte Leistung'
        else:
            label_leistung='Leistung [kW]'
        
        mess_jahresenergie, mess_verbrauch=calc_daily_energy(reference_data)
        if not normed_analysis:
            print('Jahresenergieverbrauch der Messdaten:', mess_verbrauch, ' kWh')

        bspjahr = pd.read_excel('resources/bspjahr.xlsx')
        
        ####Tagesverlauf von Mittelwert und Std.

        if plots:
            subfig, (ax1,ax2) = plt.subplots(1, 2,figsize=(12.8,7.2))
            
            ax1.plot(np.mean(tages_messdaten,axis=0),color='tab:orange')
            ax1.plot(simulation_mean,color='tab:blue')
            ax1.plot(np.max(tages_messdaten,axis=0),color='tab:orange',linestyle='dashed')
            ax1.plot(max_max_simulation,color='tab:blue',linestyle='dashed')
            ax1.legend(['Messdaten (MW)','Simulation (MW)','Messdaten (MAX)','Simulation (MAX)'])
            ax1.set_title('Mittel- und Maximalwert')
            ax1.set_ylabel(label_leistung)
            ax1.grid()
            ax1.set_xlabel('Zeit [h]')
            ax1.set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))

            ax2.plot(simulation_std)
            ax2.plot(np.std(tages_messdaten,axis=0))
            #ax2.plot(np.std(tages_messdaten,axis=0),linewidth=3,color='orange')
            ax2.legend(['Simulation','Messdaten'])
            ax2.set_title('Standardabweichung')
            ax2.set_ylabel(label_leistung)
            ax2.grid()
            ax2.set_xlabel('Zeit [h]')
            ax2.set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))
            """
            ax3.plot(np.max(tages_messdaten,axis=0),linewidth=3,color='orange')
            ax3.plot(max_max_simulation,alpha=0.9,linewidth=0.5,color='blue')
            ax3.plot(np.max(tages_messdaten,axis=0),linewidth=3,color='orange')
            ax3.legend(['Messdaten','Simulation'])
            ax3.set_title('Mittelwert')
            ax3.set_ylabel(label_leistung)
            ax3.grid()
            ax3.set_xlabel('Zeit [h]')
            ax3.set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))
            """
            if export_plots:
                plt.savefig(directory_export + 'Tagesverlauf_MW_STD_Alle_' + filename_plots + '.pdf', bbox_inches='tight')
            
        ####
        
        for k in range(0,n_runs):
            sim_data=all_sim_data[k]
            sim_data_long=simulation_matrix[:,k]

            #whole_MPE=np.sum(np.abs(sim_data_long-messdaten)/sim_data_long)*100/35040
            #print('MPE im Jahr = ', whole_MPE)

            tages_simulation=np.reshape(sim_data,[365,96])

            mean_tages = np.mean(tages_simulation,axis=1)
            std_tages = np.std(tages_simulation,axis=1)

            
            XS = 365-pd.DataFrame(tages_simulation).max(axis=1).rank().to_numpy()
            YS = pd.DataFrame(tages_simulation).max(axis=1).to_numpy()
            XM = 365-pd.DataFrame(tages_messdaten).max(axis=1).rank().to_numpy()
            YM = pd.DataFrame(tages_messdaten).max(axis=1).to_numpy()

            sim_jahresenergie, sim_verbrauch=calc_daily_energy(sim_data_long)
            #print('Jahresenergie [kWh]: Messdaten =', mess_verbrauch, ', Simulation =', sim_verbrauch, ', Differenz in Prozent =', np.abs(mess_verbrauch-sim_verbrauch)/sim_verbrauch*100)
            
            #print('Jahresenergieverbrauch der Simulation:', sim_verbrauch, ' kWh')

            statistics_energy[k] = [sim_verbrauch]

            differenz_energy=np.abs(mess_jahresenergie-sim_jahresenergie)
            MPE_Energy=differenz_energy/sim_jahresenergie*100
            #df_statistics=pd.DataFrame(MPE_Energy,columns=['Energie MPE']).describe()
            
            daily_MPE=np.zeros(365)
            daily_MPE_mean=np.zeros(365)
            daily_MPE_max=np.zeros(365)
            daily_MPE_min=np.zeros(365)

            for day in range(0,365):
                if normed_analysis:
                    daily_MPE[day]=np.sum(np.abs(tages_messdaten[day]-tages_simulation[day]))/n_steps
                else:
                    daily_MPE[day]=MPE(tages_messdaten[day],tages_simulation[day])
            
            daily_abstand_mean = np.abs(np.mean(tages_simulation,axis=1)-np.mean(tages_messdaten,axis=1))
            
            if normed_analysis:
                daily_MPE_mean=np.abs(np.mean(tages_simulation,axis=1)-np.mean(tages_messdaten,axis=1))
                daily_MPE_max=np.abs(np.max(tages_simulation,axis=1)-np.max(tages_messdaten,axis=1))
                daily_MPE_min=np.abs(np.min(tages_simulation,axis=1)-np.min(tages_messdaten,axis=1))
            else:
                daily_MPE_mean=np.abs(np.mean(tages_simulation,axis=1)-np.mean(tages_messdaten,axis=1))/np.mean(tages_messdaten,axis=1)*100
                daily_MPE_max=np.abs(np.max(tages_simulation,axis=1)-np.max(tages_messdaten,axis=1))/np.max(tages_messdaten,axis=1)*100
                daily_MPE_min=np.abs(np.min(tages_simulation,axis=1)-np.min(tages_messdaten,axis=1))/np.min(tages_messdaten,axis=1)*100

            if normed_analysis:
                daily_MPE_max_vz=(np.max(tages_simulation,axis=1)-np.max(tages_messdaten,axis=1))
                daily_MPE_min_vz=(np.min(tages_simulation,axis=1)-np.min(tages_messdaten,axis=1))
            else:
                daily_MPE_max_vz=(np.max(tages_simulation,axis=1)-np.max(tages_messdaten,axis=1))/np.max(tages_messdaten,axis=1)*100
                daily_MPE_min_vz=(np.min(tages_simulation,axis=1)-np.min(tages_messdaten,axis=1))/np.min(tages_messdaten,axis=1)*100

            statistics_describe[k]=pd.DataFrame(daily_MPE).describe()
            statistics_describe_mean[k]=pd.DataFrame(daily_MPE_mean).describe()
            statistics_describe_min[k]=pd.DataFrame(daily_MPE_min).describe()
            statistics_describe_max[k]=pd.DataFrame(daily_MPE_max).describe()
            statistics_describe_min_vz[k]=pd.DataFrame(daily_MPE_min_vz).describe()
            statistics_describe_max_vz[k]=pd.DataFrame(daily_MPE_max_vz).describe()

            statistics_index[k]=[np.argmax(daily_MPE),np.argmin(daily_MPE)]
            
            ####
            
            if plots and k==0:
                E_sim=calc_energy(tages_simulation)
                E_mess=calc_energy(tages_messdaten)
                E_delta=np.abs(E_sim-E_mess)/E_mess
                print('Simulation:', E_sim, ' [kWh], Messdaten:', E_mess, ' [kWh], MAPE:', E_delta*100, '\%')


                if normed_analysis:
                    max_power = max(np.max(reference_data), np.max(sim_data))+0.1
                    min_power = min(np.min(reference_data), np.min(sim_data))-0.1
                else:
                    max_power = max(np.max(reference_data), np.max(sim_data))+10
                    min_power = min(np.min(reference_data), np.min(sim_data))-10
            
                """
                plt.figure()
                plt.plot(simulation_mean,label='Simulation')
                plt.title('Mittelwert der Leistung pro Tag nach Simulation')
                plt.xlabel('Zeitschritt in 15 Minuten')
                plt.ylabel('Leistung [kW]')
                """

                ####Leistungen fuer jeden Zeitschritt im ganzen Jahr

                plt.rcParams.update({
                    "font.size": '32'
                })

                plt.figure(figsize=(24,4.8))
                plt.plot(sim_data_long)
                plt.ylabel(label_leistung)
                plt.ylim([min_power,max_power])
                #plt.xticks(np.arange(0,35040, step=384),np.arange(0,365, step=4))
                plt.title('Simulierte Leistung über ein Jahr')
                if export_plots:
                    plt.savefig(directory_export + 'Jahreslastgang_Simulation_' + filename_plots + '.pdf', bbox_inches='tight')

                plt.figure(figsize=(24,4.8))
                plt.plot(reference_data)
                plt.ylabel(label_leistung)
                plt.ylim([min_power,max_power])
                plt.title('Gemessene Leistung über ein Jahr')
                
                if export_plots:
                    plt.savefig(directory_export + 'Jahreslastgang_Referenz_' + filename_plots + '.pdf', bbox_inches='tight')

                plt.rcParams.update({
                    "font.size": '16'
                })

                ####Tagesverlauf von Mittelwert und Std.

                subfig, (ax1,ax2) = plt.subplots(1, 2,figsize=(12.8,4.8))
                
                ax1.plot(np.mean(tages_simulation,axis=0))
                ax1.plot(np.mean(tages_messdaten,axis=0))
                ax1.legend(['Simulation','Messdaten'])
                ax1.set_title('Mittelwert')
                ax1.set_ylabel(label_leistung)
                ax1.grid()
                ax1.set_xlabel('Zeit [h]')
                ax1.set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))

                ax2.plot(np.std(tages_simulation,axis=0),label='Simulation')
                ax2.plot(np.std(tages_messdaten,axis=0),label='Messdaten')
                ax2.legend(['Simulation','Messdaten'])
                ax2.set_title('Standardabweichung')
                ax2.set_ylabel(label_leistung)
                ax2.grid()
                ax2.set_xlabel('Zeit [h]')
                ax2.set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))

                if export_plots:
                    plt.savefig(directory_export + 'Tagesverlauf_MW_STD_' + filename_plots + '.pdf', bbox_inches='tight')

                ####                

                subfig, (ax1,ax2) = plt.subplots(1, 2,figsize=(12.8,4.8))
                
                ax1.plot(np.mean(tages_simulation,axis=1))
                ax1.plot(np.mean(tages_messdaten,axis=1))
                ax1.legend(['Simulation','Messdaten'])
                ax1.set_title('Mittelwert')
                ax1.set_ylabel(label_leistung)
                ax1.grid()
                #ax1.set_xlabel('Zeit [h]')
                #ax1.set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))

                ax2.plot(np.std(tages_simulation,axis=1))
                ax2.plot(np.std(tages_messdaten,axis=1))
                ax2.legend(['Simulation','Messdaten'])
                ax2.set_title('Standardabweichung')
                ax2.set_ylabel(label_leistung)
                ax2.grid()
                
                ###Boxplot vom Tagesverlauf

                subfig, (ax1,ax2) = plt.subplots(1, 2,figsize=(12.8,4.8))
                ax1.boxplot(tages_messdaten)
                ax1.set_title('Messdaten')
                ax1.set_ylim([min_power,max_power])
                ax1.set_xlabel('Zeit [h]')
                ax1.set_ylabel(label_leistung)
                ax1.set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))
                ax2.boxplot(tages_simulation)
                ax2.set_title('Simulation')
                ax2.set_ylim([min_power,max_power])
                ax2.set_xlabel('Zeit [h]')
                ax2.set_ylabel(label_leistung)
                ax2.set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))

                if export_plots:
                    plt.savefig(directory_export + 'Tagesverlauf_Boxplot_' + filename_plots + '.pdf', bbox_inches='tight')

                ###Punktwolke vom Tagesverlauf

                subfig, (ax1,ax2) = plt.subplots(1, 2,figsize=(14.4,4.8))
                ax1.plot(tages_messdaten.T,linewidth=0.5,color='gray')
                ax1.plot(np.mean(tages_messdaten,axis=0),linewidth=3,linestyle='--',color='black')
                ax1.plot(np.mean(tages_messdaten,axis=0)+np.std(tages_messdaten,axis=0),linewidth=3,color='black')
                ax1.plot(np.mean(tages_messdaten,axis=0)-np.std(tages_messdaten,axis=0),linewidth=3,color='black')
                ax1.set_xlabel('Zeitschritt in 15 Minuten')
                ax1.set_ylabel(label_leistung)
                ax1.set_title('Messdaten als Wolke')
                ax1.set_ylim([min_power,max_power])
                ax1.set_xlabel('Zeit [h]')
                ax1.set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))


                ax2.plot(np.reshape(simulation_matrix[:,0],[365,96]).T,linewidth=0.5,color='gray')
                ax2.plot(np.mean(tages_messdaten,axis=0),linewidth=3,linestyle='--',color='black')
                ax2.plot(np.mean(tages_messdaten,axis=0)+np.std(tages_messdaten,axis=0),linewidth=3,color='black')
                ax2.plot(np.mean(tages_messdaten,axis=0)-np.std(tages_messdaten,axis=0),linewidth=3,color='black')
                ax2.set_xlabel('Zeitschritt in 15 Minuten')
                ax2.set_ylabel(label_leistung)
                ax2.set_title('Mittelwertsband mit Simulationen')
                ax2.set_ylim([min_power,max_power])
                ax2.set_xlabel('Zeit [h]')
                ax2.set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))

                if random_plots:
                    zufallstag=np.random.randint(0,365)
                    anzahl_zufall = 5
                    for nr_zufall in range(0,anzahl_zufall):
                        zufallstag=np.random.randint(0,365)
                        
                        plt.figure()
                        plt.plot(tages_simulation[zufallstag])
                        plt.plot(tages_messdaten[zufallstag])
                        #print('AW Simulation = ', tages_simulation[zufallstag][0], ', AW Messdaten = ', tages_messdaten[zufallstag][0])
                        plt.ylabel(label_leistung)
                        plt.legend(['Simulation','Messdaten'])
                        titel='Tag ' + str(zufallstag) + ', MPE = ' + str("%1.2f"% MPE(tages_messdaten[zufallstag],tages_simulation[zufallstag])) + '%'
                        #print(hf.MPE(tages_messdaten[zufallstag],tages_simulation[zufallstag]))
                        plt.title(titel)
                
                """
                plt.figure()    
                plt.plot(MPE_Energy)
                plt.title('MPE der Energie')
                plt.xlabel('Tag im Jahr')
                plt.ylabel('Energie [%]')
                """
                """
                plt.figure()    
                plt.plot(daily_abstand_mean,label='Abstand')
                plt.plot(diff_bins,label='Groesse States')
                plt.xlabel('Tag im Jahr')
                plt.ylabel('Unnormierter Abstand')
                """

                subfig, (ax1,ax2,ax3) = plt.subplots(1, 3,figsize=(19.2,4.8))
                ax1.plot(np.min(tages_simulation,axis=1),label='Simulation')
                ax1.plot(np.min(tages_messdaten,axis=1),label='Messdaten')
                ax1.legend(['Simulation','Messdaten'])
                ax1.set_xlabel('Tag im Jahr')
                ax1.set_title('Minimale Leistung pro Tag')
                ax1.set_ylabel(label_leistung)

                ax2.plot(np.mean(tages_simulation,axis=1),label='Simulation')
                ax2.plot(np.mean(tages_messdaten,axis=1),label='Messdaten')
                ax2.legend(['Simulation','Messdaten'])
                ax2.set_xlabel('Tag im Jahr')
                ax2.set_ylabel(label_leistung)
                ax2.set_title('Mittlere Leistung pro Tag')

                ax3.plot(np.max(tages_simulation,axis=1),label='Simulation')
                ax3.plot(np.max(tages_messdaten,axis=1),label='Messdaten')
                ax3.legend(['Simulation','Messdaten'])
                ax3.set_xlabel('Tag im Jahr')
                ax3.set_ylabel(label_leistung)
                ax3.set_title('Maximale Leistung pro Tag')

                """
                subfig, (ax1,ax2,ax3) = plt.subplots(1, 3,figsize=(19.2,4.8))
                ax1.boxplot([np.min(tages_simulation,axis=1),np.min(tages_messdaten,axis=1)],labels=['Simulation','Messdaten'])
                ax1.set_xlabel('Tag im Jahr')
                ax1.set_title('Minimale Leistung pro Tag')
                ax1.set_ylabel(label_leistung)

                ax2.boxplot([np.mean(tages_simulation,axis=1),np.mean(tages_messdaten,axis=1)],labels=['Simulation','Messdaten'])
                ax2.set_xlabel('Tag im Jahr')
                ax2.set_ylabel(label_leistung)
                ax2.set_title('Mittlere Leistung pro Tag')

                ax3.boxplot([np.max(tages_simulation,axis=1),np.max(tages_messdaten,axis=1)],labels=['Simulation','Messdaten'])
                ax3.set_xlabel('Tag im Jahr')
                ax3.set_ylabel(label_leistung)
                ax3.set_title('Maximale Leistung pro Tag')
                """

                ######

                plt.figure()
                sns.boxplot(data=[[reference_data],[sim_data_long]])

                descr_sim=pd.DataFrame(reference_data).describe()
                descr_mess=pd.DataFrame(sim_data_long).describe()


                

                this_fig, this_ax=plt.subplots()
                multiind=pd.MultiIndex.from_product([['Minimalwerte','Mittelwerte','Maximalwerte'],['Simulation','Messdaten']],names=['Groesse','Art'])
                extr_val=np.array([np.min(tages_simulation,axis=1).T,np.min(tages_messdaten,axis=1).T,np.mean(tages_simulation,axis=1).T,np.mean(tages_messdaten,axis=1).T,np.max(tages_simulation,axis=1).T,np.max(tages_messdaten,axis=1).T]).T
                df_minmeanmax=pd.DataFrame(extr_val,columns=multiind).stack(level=[0,1]).reset_index(level=0, drop=True).reset_index()

                sns.boxplot(x='Groesse',y=0, data=df_minmeanmax, hue="Art",width=0.5,hue_order=['Simulation','Messdaten'],order=['Minimalwerte','Mittelwerte','Maximalwerte'])
                plt.xlabel('')
                plt.ylabel(label_leistung)
                handles, labels = this_ax.get_legend_handles_labels()
                plt.legend(handles=handles[0:], labels=labels[0:])

                if export_plots:
                    plt.savefig(directory_export + 'Boxplot_Min_Max_MW_' + filename_plots + '.pdf', bbox_inches='tight')

                ###Jahresdauerlinie maximale Leistung

                plt.figure()
                #plt.bar(XS,YS,alpha=0.5)
                #plt.bar(XM,YM,alpha=0.5)
                plt.scatter(XS,YS,label='Simulation',s=2)
                plt.scatter(XM,YM,label='Messdaten',s=2)
                #plt.title('Tägliche Maximalwerte absteigend sortiert')
                plt.xlabel('Rang')
                plt.ylabel(label_leistung)
                plt.legend(['Simulation','Messdaten'])
                
                if export_plots:
                    plt.savefig(directory_export + 'Geordnete_Maximalleistungen_' + filename_plots + '.pdf', bbox_inches='tight')

                ######
                
                plt.rcParams.update({
                    "font.size": '24'
                })

                subfig, axs = plt.subplots(1, 3,figsize=(19.2,4.8))
                nr=0
                axs[nr].hist(np.min(tages_simulation,axis=1),bins=50,density=True,alpha=0.7)
                axs[nr].hist(np.min(tages_messdaten,axis=1),bins=50,density=True,alpha=0.7)
                axs[nr].set_title('Minimale Leistung pro Tag')
                axs[nr].legend(['Simulation','Messdaten'])
                axs[nr].set_ylabel('Häufigkeitsdichte')

                nr=1
                axs[nr].hist(np.mean(tages_simulation,axis=1),bins=50,density=True,alpha=0.7)
                axs[nr].hist(np.mean(tages_messdaten,axis=1),bins=50,density=True,alpha=0.7)
                axs[nr].set_title('Mittlere Leistung pro Tag')
                axs[nr].legend(['Simulation','Messdaten'])

                nr=2
                axs[nr].hist(np.max(tages_simulation,axis=1),bins=50,density=True,alpha=0.7)
                axs[nr].hist(np.max(tages_messdaten,axis=1),bins=50,density=True,alpha=0.7)
                axs[nr].set_title('Maximale Leistung pro Tag')
                axs[nr].legend(['Simulation','Messdaten'])
                
                maxprob=np.zeros(3)
                minprob=np.zeros(3)

                for i in [0,1,2]:
                    minprob[i], maxprob[i]=axs[i].get_ylim()

                maxmaxprob=np.max(maxprob)
                minminprob=np.min(minprob)

                for i in range(0,3):
                    #axs[i].set_xlim([min_power, max_power])
                    axs[i].set_ylim([minminprob,maxmaxprob])
                    axs[i].set_xlabel(label_leistung)
                    #axs[i].set_ylabel('Häufigkeitsdichte')

                if export_plots:
                    plt.savefig(directory_export + 'Histogramm_Min_Max_MW_' + filename_plots + '.pdf', bbox_inches='tight')
                
                plt.rcParams.update({
                    "font.size": '16'
                })
                """
                subfig, axs = plt.subplots(2, 3,figsize=(19.2,9.6))
                axs[0,0].hist(np.min(tages_simulation,axis=1),bins=50,density=True,alpha=0.3)
                axs[0,0].hist(np.min(tages_messdaten,axis=1),bins=50,density=True,alpha=0.3)
                axs[0,0].set_title('Minimale Leistung pro Tag')
                axs[0,0].legend(['Simulation','Messdaten'])

                axs[0,1].hist(np.mean(tages_simulation,axis=1),bins=50,density=True,alpha=0.3)
                axs[0,1].hist(np.mean(tages_messdaten,axis=1),bins=50,density=True,alpha=0.3)
                axs[0,1].set_title('Mittlere Leistung pro Tag')

                axs[0,2].hist(np.max(tages_simulation,axis=1),bins=50,density=True)
                axs[0,2].hist(np.max(tages_messdaten,axis=1),bins=50,density=True)
                axs[0,2].set_title('Maximale Leistung pro Tag')

                maxprob=np.zeros([2,3])
                minprob=np.zeros([2,3])

                for i in [0,1]:
                    for j in [0,1,2]:
                        minprob[i,j], maxprob[i,j]=axs[i,j].get_ylim()

                maxmaxprob=np.max(maxprob)
                minminprob=np.min(minprob)

                for i in range(0,2):
                    for j in range(0,3):
                        axs[i,j].set_xlim([min_power, max_power])
                        axs[i,j].set_ylim([minminprob,maxmaxprob])

                axs[1,0].set_xlabel(label_leistung)
                axs[1,1].set_xlabel(label_leistung)
                axs[1,2].set_xlabel(label_leistung)
                axs[0,0].set_ylabel('Relative Häufigkeit')
                axs[1,0].set_ylabel('Relative Häufigkeit')
                """

                subfig, axs = plt.subplots(1, 2,figsize=(14.4,4.8))
                nr=0
                axs[nr].hist(simulation_matrix[:,k],bins=100,density=True)
                axs[nr].set_xlim([min_power,max_power])
                axs[nr].set_title('Histogramm der simulierten Leistungen')
                axs[nr].set_xlabel(label_leistung)
                axs[nr].set_ylabel('Häufigkeitsdichte')
                
                nr=1
                axs[nr].hist(reference_data,bins=100,density=True)
                axs[nr].set_xlim([min_power,max_power])
                axs[nr].set_title('Histogramm der gemessenen Leistungen')
                axs[nr].set_xlabel(label_leistung)
                axs[nr].set_ylabel('Häufigkeitsdichte')

                
                for i in [0,1]:
                    minprob[i], maxprob[i]=axs[i].get_ylim()

                maxmaxprob=np.max(maxprob)
                minminprob=np.min(minprob)

                for i in [0,1]:
                    axs[i].set_ylim([minminprob,maxmaxprob])

                if export_plots:
                    plt.savefig(directory_export + 'Histogramm_Jahresleistung_' + filename_plots + '.pdf', bbox_inches='tight')

                ######

                """
                plt.figure()
                plt.plot(daily_MPE)
                plt.title('Täglicher MPE in Prozent')
                plt.xlabel('Tag im Jahr')
                """

                ######

                """
                subfig, (ax1,ax2,ax3) = plt.subplots(1, 3,figsize=(24,4.8))

                ax1.plot(daily_MPE_min)
                ax1.set_title('Täglicher MPE der minimalen Leistung')
                ax1.set_ylabel('MPE [%]')
                ax1.set_xlabel('Tag im Jahr')

                ax2.plot(daily_MPE_mean)
                ax2.set_title('Täglicher MPE der mittleren Leistung')
                ax2.set_ylabel('MPE [%]')
                ax2.set_xlabel('Tag im Jahr')

                ax3.plot(daily_MPE_max)
                ax3.set_title('Täglicher MPE der maximalen Leistung')
                ax3.set_ylabel('MPE [%]')
                ax3.set_xlabel('Tag im Jahr')

                """

                """

                plt.figure()
                plt.hist(daily_MPE,bins=100)
                plt.title('Histogramm der MPEs der Leistung')

                fig=fig+1
                plt.figure(fig)
                plt.plot(daily_MPE_max)
                plt.title('Täglicher MPE der Maximalwerte in Prozent')
                plt.xlabel('Tag im Jahr')

                fig=fig+1
                plt.figure(fig)
                plt.plot(daily_MPE_min)
                plt.title('Täglicher MPE der Minimalwerte in Prozent')
                plt.xlabel('Tag im Jahr')
                """
                
                #####
                
                plt.rcParams.update({
                    "font.size": '24'
                })

                subfig, (ax1,ax2) = plt.subplots(1, 2,figsize=(24,4.8))
                
                if normed_analysis:
                    sns.boxplot(data=[daily_MPE,daily_MPE_mean,daily_MPE_max,daily_MPE_min],ax=ax1,width=0.5)     
                    ax1.set_xticks([0,1,2,3],['MAPE pro Tag','Mittelwert','Maximalwerte','Minimalwerte'])
                else:
                    sns.boxplot(data=[daily_MPE,daily_MPE_max,daily_MPE_min,MPE_Energy],ax=ax1,width=0.5)     
                    ax1.set_xticks([0,1,2,3],['MAPE pro Tag','Maximalwerte','Minimalwerte','Energie'])
                ax1.set_title('MAPE')
                ax1.set_ylabel('MAPE in [\%]')

                ax2.axhline(y=0,color='black')
                sns.boxplot(data=[daily_MPE_max_vz,daily_MPE_min_vz],ax=ax2,width=0.25)  
                ax2.set_xticks([0,1],['Maximalwerte','Minimalwerte'])
                ax2.set_title('MPE')
                ax2.set_ylabel('MPE in [\%]')

                zufallstag=np.argmax(daily_MPE)
                
                if export_plots:
                    plt.savefig(directory_export + 'Boxplot_MAPE_Min_Max_MW_' + filename_plots + '.pdf', bbox_inches='tight')

                plt.rcParams.update({
                    "font.size": '16'
                })

                ###Maximaler und minimaler MAPE pro Jahr

                subfig, (ax1,ax2) = plt.subplots(1, 2,figsize=(19.2,4.8))

                ax1.plot(tages_simulation[zufallstag])
                ax1.plot(tages_messdaten[zufallstag])
                ax1.set_ylabel(label_leistung)
                ax1.legend(['Simulation','Messdaten'])
                if normed_analysis:
                    titel='Maximaler Jahres-MPE an ' + str(zufallstag) + ' (Typtag: ' + str(bspjahr['Tag'].iloc[zufallstag]) + '), MPE = ' + str("%1.2f"% daily_MPE[zufallstag])
                else:
                    titel='Maximaler Jahres-MPE an ' + str(zufallstag) + ' (Typtag: ' + str(bspjahr['Tag'].iloc[zufallstag]) + '), MPE = ' + str("%1.2f"% daily_MPE[zufallstag]) + '%'
                ax1.set_title(titel)
                ax1.set_xlabel('Zeit [h]')
                ax1.set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))
                ax1.grid()

                zufallstag=np.argmin(daily_MPE)
                
                ax2.plot(tages_simulation[zufallstag])
                ax2.plot(tages_messdaten[zufallstag])
                ax2.set_ylabel(label_leistung)
                ax2.legend(['Simulation','Messdaten'])
                if normed_analysis:
                    titel='Minimaler Jahres-MPE an ' + str(zufallstag) +  ' (Typtag: ' + str(bspjahr['Tag'].iloc[zufallstag]) + '), MPE = ' + str("%1.2f"% daily_MPE[zufallstag])            
                else:
                    titel='Minimaler Jahres-MPE an ' + str(zufallstag) +  ' (Typtag: ' + str(bspjahr['Tag'].iloc[zufallstag]) + '), MPE = ' + str("%1.2f"% daily_MPE[zufallstag]) + '%'
                ax2.set_title(titel)
                ax2.set_xlabel('Zeit [h]')
                ax2.set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))
                ax2.grid()

                ######

                subfig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8))

                maxwert=np.max([np.max(tages_simulation),np.max(tages_messdaten)])
                minwert=np.min([np.min(tages_simulation),np.min(tages_messdaten)])

                sns.heatmap(tages_simulation,vmin=minwert,vmax=maxwert,ax=axs[0],cbar_kws={'label': label_leistung}).set(title='Simulation')
                axs[0].set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))
                axs[0].set_xlabel('Zeit am Tag [h]')
                axs[0].set_ylabel('Tag im Jahr')

                sns.heatmap(tages_messdaten,vmin=minwert,vmax=maxwert,ax=axs[1],cbar_kws={'label': label_leistung}).set(title='Messdaten')
                axs[1].set_xticks(np.arange(0,108, step=12),np.arange(0,25, step=3))
                axs[1].set_xlabel('Zeit am Tag [h]')

                if export_plots:
                    plt.savefig(directory_export + 'Heatmap_Simulation_Messdaten_' + filename_plots + '.pdf', bbox_inches='tight')


        if show_tables:
            print('Mittlerer MPE tageweise:')
            display(statistics_describe)
            print('MPE der mittleren Leistung:')
            display(statistics_describe_mean)
            print('MPE der Maximalleistung:')
            display(statistics_describe_max)
            print('MPE der Minimalleistung:')
            display(statistics_describe_min)
            print('MPE der Maximalleistung mit Vorzeichen:')
            display(statistics_describe_max_vz)
            print('MPE der Minimalleistung mit Vorzeichen:')
            display(statistics_describe_min_vz)
            #print('Maximaler minimaler MPE = ', statistics_describe.max(axis=1)['min'])
            #print('Minimaler maximaler MPE = ', statistics_describe.min(axis=1)['max'])
            #print('Mittlerer MPE = ', statistics_describe.mean(axis=1)['mean'])
            display(statistics_index)
            display(statistics_energy)

        MAPEs=[MAPE_mean, MAPE_std, MAPE_max]
        return MAPEs, reference_data, sim_data_long, daily_MPE


