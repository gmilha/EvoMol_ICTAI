import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from cycler import cycler
import os
from os.path import dirname, join
import csv

##########################################################################################################
#################### All run management : compile mean, median, std or pooling all the run ###############
####################                   Management of Step file                      ######################
##########################################################################################################

def save_stat_all_run(stat, headers, file_name):
    """saving generated statistics to csv file

    Args:
        stat (_type_): _description_
        headers (_type_): _description_
        file_name (text): path and csv file name
    """
    #print(stat)
    Dstat_df = pd.DataFrame(stat, columns=headers)
    Dstat_df.to_csv(file_name, index=False)

##### File steps.csv contains results of each steps like objective function score, n_discarded_tabu; n_discarded_silly_walk; time_stamp... 
def run_mean_std(path, nb_run, file="steps.csv"):
    """
    all run use and success mean, median and standard error calculations
    Args:
        path : path to folder containing results several runs with a same config
        runpath : path to folder of a specific run
        nb_run : nb of run to consider
        file : name of the file to consider (default : "steps.csv")
    """
    # Initialising list to keep nb_run files data
    all_data = []

    # Loading nb_run CSV files
    for i in range(1, nb_run+1):  
        #print("run"+str(i))
        #file_path = path+runpath+"_run"+str(i)+"/"+file
        file_path = path+"run"+str(i)+"/"+file
        
        df = pd.read_csv(file_path)  
    # Headers keeping:
        if i==1: 
             headers = pd.read_csv(file_path).columns
    # Convert data from string to float
        df = df.apply(pd.to_numeric, errors='coerce')  # Remplace les strings par des NaN si non convertible
        all_data.append(df)
    j=1
    for df in all_data:
         print(j)
         print(df.values.shape)
         j+=1
    # Stack all DataFrames in one to compute statistics
    stacked_data = np.stack([df.values for df in all_data], axis=2)

    # mean, std and media, computation on step axis (axis=2)
    sum =  np.nansum(stacked_data, axis=2)  # sum of all run
    mean = np.nanmean(stacked_data, axis=2)  # Mean of all run
    std = np.nanstd(stacked_data, axis=2)    # std of all run
    median = np.nanmedian(stacked_data, axis=2)    #  Median of all run

    save_stat_all_run(sum, headers, path+'sum_'+file)
    save_stat_all_run(mean, headers, path+'mean_'+file)
    save_stat_all_run(std, headers, path+'std_'+file)
    save_stat_all_run(median, headers, path+'median_'+file)

##########################################################################################################
####################                Management of Use and Success files             ######################
##########################################################################################################
def loadResults(feature, ql_path, action):
    """
    Loading results from csv file in a dict per action and feature combination
    """
    try:
        filename = ql_path + action+"_"+feature+".csv" #ql_path + "/"+action+"_"+feature+".csv"
        with open(filename, mode='r', newline='', encoding='utf-8') as file:
            final_feat = next(csv.DictReader(file))
        print(f"{ql_path}{action}_{feature} loaded successfully.")
        
        if 'Step' in final_feat:
             del final_feat['Step']
        # if needed, keeping only the last line
        #final_feat = feat_reader.iloc[-1, 1:]
        #print("Nb ",feature, " : ",len(final_feat)) 
        return final_feat
    
    except FileNotFoundError:
        print(f"File {ql_path}{action}_{feature}.csv not found.")
    except Exception as e:
        print(f"An error occurred while loading {ql_path}{action}_{feature}.csv: {e}")

def SaveDictCSV(csvfile, dict):
    """
    Saving a dict in a csv file 

    Args:
        csvfile (text): path and csv file name
        dict (dict): dictionary to save
    """
    with open(csvfile, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=dict.keys())
        writer.writeheader()
        writer.writerow(dict)

def SuccessRate(success, use_run_i,filename):
    """
    Compute success rate for used indexes

    Args:
        success (dict): count of success per index
        use_run_i (dict): count of use per index
        filename (text) : path and csv file name
  """
    #for i, u in enumerate(use):
    Success_rate = {}
    for key, value in success.items():
        if int(use_run_i[key]) > 0:
            Success_rate[key] = round(int(value) / int(use_run_i[key]),2)
        #else : print("non used index :", key)
    #print("SuccessRate :", Success_rate)
    #Saving Success_rate.csv
    SaveDictCSV(filename, Success_rate)
    
    return Success_rate

def ManageSuccessRate(feature,use_all_run,run_id, curr_run, ql_path, action):
    if feature == "use":
        use_all_run[run_id] = curr_run
       #print("all_use : ", use_all_run[run_id].keys())
    if feature == "success":
        SuccessRate(curr_run, use_all_run[run_id], ql_path+action+"_"+feature+"Rate.csv")
        #Success_rate = SuccessRate(curr_run, use_all_run[run_id])        
    return use_all_run

###### Files AddA, ChB, RmA.csv:  sum of all 10 run 
def SumAllRun(path, runpath, nb_run, actionsList):
    """
    Summing results of all runs
    :param path : path to folder containing results of all the stochastic Qlearning runs
    :param runpath : folder to consider containing results of each stochastic Qlearning runs
    """
    for action in actionsList:#["addA", "rmA", "chB"]:
        use_all_run = {}
        for feature in ["use", "success"]:#, "success_rates"] :
                W_sum = {}
                Total_W = {}
               
                for i in range(1, nb_run+1):
                        #print("run"+str(i))
                        ql_path = path + "run"+str(i)+"/"  #path + runpath+"_run"+str(i)+"/"  #
                        #print(ql_path + "/"+action+"_"+feature+".csv")
                        curr_run = loadResults(feature, ql_path, action)

                        use_all_run = ManageSuccessRate(feature,use_all_run,f"run{i}",curr_run, ql_path, action)
                        
                        for key, value in curr_run.items():
                                if key in W_sum: 
                                    W_sum[key] += int(value) #float(value)
                                else:
                                    W_sum[key] = int(value) #float(value)
                                  
                #Sort ecpf indexes in the dict
                all_positions = sorted(set(int(pos) for pos in W_sum.keys()))#W_sum[feature].keys()))
                for i, position in enumerate(all_positions):
                     #print(i, position)
                    Total_W[str(position)] = W_sum[str(position)]
                #print(Total_W)
                SaveDictCSV(path+action+"_"+feature+".csv", Total_W)
                use_all_run = ManageSuccessRate(feature, use_all_run,"Total", Total_W, path, action)
                                
       ### Diag_cdf(path, action, wused=True)
######################################################################################

######################################################################################
################### Definition of Different plots configuration ######################
######################################################################################
    
def PlotOverSteps(ql_path, config, time, ql_steps_reader, rdm_steps_reader, data, title, xlabel, ylabel, ecfp, stat, div_nbstep):
    """
    Plotting over step (used for timestamp, tabu list fail or silly walk failed)
    :param ql_path : path to folder containing results of stochastic Qlearning runs
    :param ql_file : name of the file to consider containing results of stochastic Qlearning runs
    :param time : if drawing time, needed to divide the data by 60
    :param ql_steps_reader : path and file to consider for stochastic Qlearning runs
    :param rdm_steps_reader : path and file to consider for random runs
    :param data : feature to plot ( 'timestamps', 'n_discarded_tabu', 'n_discarded_sillywalks', )
    :param title : graph title
    :param xlabel : label of horizontal axis 
    :param ylabel : label of vertical axis
    :param efcp : ecfp even number considered for stochastic Qlearning runs
    :param stat : stat to perform (mean, median)
    :param div_nbstep : diviser used to limit the nb of steps considered as one point in graphs
    """
    ql_data = np.array(ql_steps_reader[data].tolist()) / time
    rdm_data = np.array(rdm_steps_reader[data].tolist()) / time
    
    #div_nbstep = 100 #divide the len of each file by this value  
    #permet de générer des graphiques plus lisibles : 
    # if div_nbstep =1000 = graphes plots each steps (if steps = 1000) => too much info, difficult to read
    # if div_nbstep =100 = graphes plots per 10 steps (if steps = 1000)
    # if div_nbstep =10 = graphes plots per 100 steps (if steps = 1000) => too smooth

    # Execution time evolution over steps
    plt.figure(stat+" "+title+" by " + str(int(np.floor(len(ql_data) / div_nbstep))) + " steps ", figsize=(16, 9))
    plt.title(stat+" "+title+" by " + str(int(np.floor(len(ql_data) / div_nbstep))) + " steps " + config)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if div_nbstep>1:
        ql_mean = [ql_data[i : i + int(np.floor(len(ql_data) / div_nbstep))].mean() for i in range(0, len(ql_data), int(np.floor(len(ql_data) / div_nbstep)))]
        rdm_mean = [rdm_data[i : i + int(np.floor(len(rdm_data) / div_nbstep))].mean() for i in range(0, len(rdm_data), int(np.floor(len(rdm_data) / div_nbstep)))]
        plt.plot(np.array(range(0, len(ql_data), int(np.floor(len(ql_data) / div_nbstep)))), ql_mean, color='dodgerblue', label="Q-Learning")
        plt.plot(np.array(range(0, len(rdm_data), int(np.floor(len(rdm_data) / div_nbstep)))), rdm_mean, color='firebrick', label="Random")
        ql_total_mean = np.array(ql_data[1:]).mean()
        rdm_total_mean = np.array(rdm_data[1:]).mean()
        plt.plot(np.array(range(0, len(ql_data), int(np.floor(len(ql_data) / div_nbstep)))), [ql_total_mean] * len(ql_mean), color='dodgerblue', linestyle='--', label=stat+" Q-Learning")
        plt.plot(np.array(range(0, len(rdm_data), int(np.floor(len(rdm_data) / div_nbstep)))), [rdm_total_mean] * len(rdm_mean), color='firebrick', linestyle='--', label=stat+" Random")
        plt.annotate(int(np.ceil(ql_total_mean)), xy=(1, ql_total_mean), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points', fontsize=16)
        plt.annotate(int(np.ceil(rdm_total_mean)), xy=(1, rdm_total_mean), xytext=(8, 0),
                    xycoords=('axes fraction', 'data'), textcoords='offset points', fontsize=16)

    else: 
        plt.plot(ql_data, label="Q-Learning", color='dodgerblue')
        plt.plot(rdm_data, label="Random", color='firebrick')
        plt.annotate('%0.2f' % ql_data[-1], xy=(1, ql_data[-1]), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
        plt.annotate('%0.2f' % rdm_data[-1], xy=(1, rdm_data[-1]), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')

    #plt.fill_between(np.array(range(len(ql_data))), ql_data, 0, color='dodgerblue', alpha=.25)
    #plt.fill_between(np.array(range(len(rdm_data))), rdm_data, 0, color='firebrick', alpha=.25)

    plt.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.75,
            alpha=0.2)

    plt.xlim([0, len(ql_data)])
    
    plt.ylim([0, max(max(ql_data), max(rdm_data)) + 5])

    plt.legend()
    plt.tight_layout()
    plt.savefig(ql_path + "/"+data+"_" + str(int(np.floor(len(ql_data) / div_nbstep))) + "steps_efcp"+str(ecfp)+"_"+stat+".png")
    plt.close()


######################################################################################
##################### Plotting tabu list fail and silly walk failed ######################
######################################################################################
def plotfail(ql_path, ql_file, rdm_file, config, ecfp, stat, div_nbstep):
    """
    Plotting tabu list fail and silly walk failed
    :param ql_path : path to folder containing results of stochastic Qlearning runs
    :param ql_file : name of the file to consider containing results of stochastic Qlearning runs
    :param rdm_file : name of the file to consider containing results of random runs
    :param div_nbstep : diviser used to limit the nb of steps considered as one point in graphs
    :param efcp : ecfp even number considered for stochastic Qlearning runs
    :param stat : stat to perform (mean, median)
    """
 
    ql_steps_reader = pd.read_csv(ql_path + ql_file)
    print(ql_path + ql_file)
    rdm_steps_reader = pd.read_csv(rdm_file)
    print(rdm_file)

    # Execution time evolution over steps
    PlotOverSteps(ql_path, config, 60, ql_steps_reader, rdm_steps_reader,'timestamps',"Temps d'exécution ", "Steps" ,"Time (min)", ecfp, stat, 1)
#    PlotOverSteps(ql_path, config, 1, ql_steps_reader, rdm_steps_reader,'n_not_improvers',"Nombre de non amélioration ","Steps" ,"Failure frequency", ecfp, stat, 1)
#    PlotOverSteps(ql_path, config, 1, ql_steps_reader, rdm_steps_reader,'n_not_improvers',stat+"s Nombre de non amélioration ","Steps" ,"Failure frequency", ecfp, stat, div_nbstep)    
    PlotOverSteps(ql_path, config, 1, ql_steps_reader, rdm_steps_reader,'total_mean',"Nombre de non amélioration ","Steps" ,"Failure frequency", ecfp, stat, 1)
    PlotOverSteps(ql_path, config, 1, ql_steps_reader, rdm_steps_reader,'n_discarded_tabu',"Nombre de tabu fails ","Steps" ,"Failure frequency", ecfp, stat, 1)
    PlotOverSteps(ql_path, config, 1, ql_steps_reader, rdm_steps_reader,'n_discarded_tabu',stat+"s Nombre de tabu fails ","Steps" ,"Failure frequency", ecfp, stat, div_nbstep)
    PlotOverSteps(ql_path, config, 1, ql_steps_reader, rdm_steps_reader,'n_discarded_sillywalks',"Nombre de molécules ne passant pas le filtre des silly walks ","Steps" ,"Failure frequency", ecfp, stat, 1)
    PlotOverSteps(ql_path, config, 1, ql_steps_reader, rdm_steps_reader,'n_discarded_sillywalks',stat+"s Nombre de molécules ne passant pas le filtre des silly walks ","Steps" ,"Failure frequency", ecfp, stat, div_nbstep)
                
##################################
### Stack Diagram per context  ###
##################################
def addlabels(x,y,value):
    for i in range(len(x)):
        plt.text(i,y[i],value[i])

def Stack_Diagram(ql_path, action, context, succes, echecs, title):
        
        fig, ax = plt.subplots(figsize=(10, 6))

        # Tracer les barres pour les succès en vert
        ax.bar(context , succes, label='Success', color='green')
        addlabels(context, succes, succes)

        # Tracer les barres pour les échecs en rouge empilées au-dessus des succès
        ax.bar(context , echecs, bottom=succes, label='Failure', color='red')
        addlabels(context, echecs+succes, echecs)

        # Ajouter des légendes et des labels
        ax.set_xlabel('context')
        ax.set_ylabel('Effectifs')
        ax.set_title('Stack bar Diagram success, failure of '+title)
        ax.legend()
        # ax.set_xticklabels(context, rotation=90) # Rotation des labels si nécessaire

        # Ajuster l'espacement
        plt.tight_layout()

        plt.savefig(ql_path + "/"+action+"_diagramme_empilé_"+title+".png")
        plt.close()

  
###########################
### Diagram per context ###
###########################

def diagram(path, action, context, success, failure, ylabel, title):
        # Largeur des barres
        bar_width = 0.35
        # Position des barres sur l'axe X
        x = np.arange(len(context))

        # Tracer les barres
        fig, ax = plt.subplots(figsize=(10, 6))

        # Barres pour les succès
        bars1 = ax.bar(x - bar_width / 2, success, bar_width, label='Success', color='green')
        #ax.bar(context , succes, label='Succès', color='green')
        # Ajout des titres et des labels
        ax.set_xlabel('context')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(context)
        ax.legend()

            # Add values up to bars
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2,  bar.get_height(),  bar.get_height(), va='bottom')  # va : vertical alignment

        #if failure:
        # Failure bars
        bars2 = ax.bar(x + bar_width / 2, failure, bar_width, label='Failure', color='r')
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2,  bar.get_height(),  bar.get_height(), va='bottom')

        # Save the graph
        plt.tight_layout()
        plt.savefig(path+"/Diagram_"+ylabel+"_"+action+".png")
        plt.close()  
        
def HbarDiagram(path, action, context, var1, var2, var1label, ylabel, title):
        # Largeur des barres
        bar_width = 0.35
        # Position des barres sur l'axe X
        y = np.arange(len(context))

        # Tracer les barres
        fig, ay= plt.subplots(figsize=(10, 6))

        # Barres pour les succès
        ay.barh(y - bar_width / 2, var1, bar_width, label=var1label, color='green')
        #ax.bar(context , succes, label='Succès', color='green')
        # Ajout des titres et des labels
        ay.set_xlabel(ylabel)
        ay.set_ylabel('context')
        ay.set_title(title)
        ay.set_yticks(y)
        ay.set_yticklabels(context)
        ay.legend()

        if var2 is not None:
        # Failure bar 
            ay.barh(y + bar_width / 2, var2, bar_width, label='Failure', color='r')

        # Save the graph
        plt.tight_layout()
        plt.savefig(path+"/Hbar_Diagram_"+ylabel+"_"+action+".png")
        plt.close()  

def CumDiagram(path, config, x, y, xcum, cum, xlabel, ylabel, Width, Align):                      
        ##### Graphique contenant les taux de succès et la courbe de fréquence
        # Tracer la fonction de répartition
        fig, ax1 = plt.subplots(figsize=(14, 7))
        ax1.bar(x, y, width=Width, align=Align, color='skyblue', edgecolor='black', alpha=0.7, label=ylabel)
        ax1.set_xlabel(xlabel, fontweight='bold')
        ax1.set_ylabel(ylabel, fontweight='bold', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')
        #ax1.set_xticklabels(x, rotation=90)

            # Ajouter la courbe cumulative sur un axe secondaire
        ax2 = ax1.twinx()
        ax2.plot(xcum, cum, color='red', marker='o', linestyle='-', label='Cumul')
        ax2.set_ylabel('Fréquence cumulée', fontweight='bold', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Synchronisation des échelles des deux axes y
        ylim_ax1 = ax1.get_ylim()
        ylim_ax2 = ax2.get_ylim()

# Fixer les limites des deux axes à la plus grande étendue
        common_ylim = (0, max(ylim_ax1[1], ylim_ax2[1]))
        ax1.set_ylim(common_ylim)
        ax2.set_ylim(common_ylim)

# Afficher le graphique
        plt.title(ylabel+" distribution per "+xlabel+" & cdf - "+config, fontweight='bold')
        fig.tight_layout()
        plt.savefig(path + "Fonction_repartion_"+ylabel+".png")
        plt.close()      
                        
def Diag_cdf(path, config, success_rate, action):#, wused=True):
        """
        Drawing distribution of success
        :param path : path to folder containing results of all the stochastic Qlearning runs
        :param ql_file : folder to consider containing results of each stochastic Qlearning runs
        """
        #df = pd.read_csv(path+action+"_success_rates.csv")#df = pd.read_csv(path+'Sum_'+action+"_success_rates.csv")
        #success_rate = df.iloc[-1, 1:]/10 #Divided by 10 due to percent sum of 10 runs
        #success_rate = pd.to_numeric(success_rate, errors='coerce')
        
        #Select used contexts only
        #if wused:
        #    use_reader= pd.read_csv(path+action+"_use.csv")#use_reader= pd.read_csv(path+'Sum_'+action+"_use.csv")
        #    final_uses = use_reader.iloc[-1, 1:]
        #    num_uses = pd.to_numeric(final_uses, errors='coerce')
        #    modality_mask = num_uses > 0
        #    success_rate = success_rate[modality_mask]
        
        ###Distribution et cumulative distribution frequency per context         
        Success_sorted = success_rate.sort_values()
        success = Success_sorted.values
        cumulative = np.cumsum(success) / np.sum(success) # Calcul de la fréquence cumulée
        labels = list(Success_sorted.index)
        CumDiagram(path, config, labels, success, labels, cumulative, 'Contexte', 'Success Rate_'+action, 0,'center')
        
        ### Histogramme et courbe cumulée des contextes pour la fréquence de taux de succès mis en classe de 0.1 pas
        bin_edges = np.linspace(0, 1, 11)  # Mise en classe du taux de succès par pas de 0.1
        count, succ_rate = np.histogram(success, bins=bin_edges) 
        Rel_freq=count/sum(count) #relative frequency calculation 
        cumulative = np.append(0,np.cumsum(Rel_freq)) # Calcul de la fréquence cumulée; ajout de la fréquence initiale 0
        #draxing histogram
        CumDiagram(path, config, succ_rate[:-1], Rel_freq, succ_rate, cumulative, 'Success Rate_'+action, 'Relative frequency of contexts_'+action, np.diff(succ_rate), 'edge')


def GraphOverSteps(title, ql_path, ql_file, csvFile, actionsList):
            color = ['dodgerblue', 'firebrick', 'forestgreen']
            plt.figure(title, figsize=(16, 9))
            plt.title(title + ql_file)

            plt.xlabel("Steps")
            plt.ylabel("Moyenne des poids")
            c = 0
            maxi = 0
            for action in actionsList:#["addA", "rmA", "chB"]:
                    reader = pd.read_csv(ql_path + "/"+action+csvFile+".csv")
                    w_mean = [np.array([reader.iloc[i][1:].tolist()]).mean() for i in range(reader.shape[0] - 1)]
            
                    plt.plot(range(reader.shape[0] - 1), [np.array([reader.iloc[i][1:].tolist()]).mean() for i in range(reader.shape[0] - 1)],
                            label=action, color=color[c])
                    c+=1
                    #print("maxi : ", maxi)
                    #print("max ", action," : ", max(w_mean))
                    if max(w_mean)> maxi:
                        maxi = max(w_mean)
                        #print("maxi ", action," : ", maxi)
                        
            plt.xlim([0, reader.shape[0] - 1])
            plt.ylim([0, maxi + 0.05])
            # plt.ylim([min([min(addA_w_mean), min(rmA_w_mean), min(chB_w_mean)]), max([max(addA_w_mean), max(rmA_w_mean), max(chB_w_mean)]) + 0.01])
            plt.grid(visible=True, color='grey', linestyle='-.', linewidth=0.75, alpha=0.2)

            plt.legend()
            plt.tight_layout()
            plt.savefig(ql_path + "/mean"+csvFile+".png")
            plt.close()


def Succ_Fail(final_success, final_uses, Relative): #, wused):
        """
        Calculating success and failure frequency or relative_frequency
        :param success_reader : success frequency or rate per context resulting from qlearning_stochastic run
        :param use_reader : use frequency per context resulting from qlearning_stochastic run
        :param Relative : whether to use relative_Frequency (percent) or Frequency (count)
        :param wused: whether to filter on contexts used only
    
        return success, failure, context
        """        
        uses = pd.to_numeric(final_uses, errors='coerce')
        if Relative:    #Relative_frequency (percent)
               # final_succes_rate = success_reader.iloc[-1, 1:]
                success = pd.to_numeric(final_success, errors='coerce')
                failure = 1 - success
                
        else :# Frequency (number; count)        
                #all_succes = success_reader.iloc[-1, 1:]
                success = pd.to_numeric(final_success, errors='coerce')
                failure = uses - success

        # filtering only used context
       # if wused:
            # Sélectionner les contextes rencontrés
        #    modality_mask = uses > 0
        #    success = success[modality_mask]
        #    failure = failure[modality_mask]
        return success, failure
    
def SuccessRatePlot(ql_path, path, config, action, use_per_action):
            used_success = loadResults("success", ql_path, action)
            #final_success = loadResults("success", ql_path, action)
            used_uses = loadResults("use", ql_path, action)
            #final_uses = loadResults("use", ql_path, action)
            
            #Selection of used contexts only
            #modality_mask = SelectUsedContext(final_uses)
            #used_success = final_success[modality_mask] 
            #used_uses = final_uses[modality_mask]
            used_success_rate = SuccessRate(used_success, used_uses,ql_path+action+"_success_rates.csv") 
            
            use_per_action[action] = (used_uses,len(used_uses))
            print("use_per_action",use_per_action)
            #Saving reference of used contexts
            used_context = used_uses.index.tolist()
            #print("Nb used success rate : ",len(used_success_rate))
            #print("used_success_rate : ",used_success_rate)
            
            #print("Nb used context : ",len(context))
            #print("Used context : ",context)
            
            #SuccessRate Plot
            used_success_rate.plot(kind='barh',rot=0,yticks=used_success_rate)            
            Diag_cdf(path, config, used_success_rate, action)#, wused=True)
            
            # Relative frequency success, failure diagramm per context 
            success_rate, failure_rate = Succ_Fail(used_success_rate, used_uses, True)#, True)
            diagram(ql_path, action, used_context, success_rate, failure_rate, "Relative frequency", config+"_all_context_rel_freq")
            HbarDiagram(ql_path, action, used_context, success_rate, failure_rate, "Success rate", "Relative frequency", config+"_all_context_rel_freq")
            
            # Stacked Bar Chart success, failure per context
            success, failure = Succ_Fail(used_success, used_uses, False)#, True)
            Stack_Diagram(ql_path, action, used_context, success, failure, config+"_all_context_freq")
            
            # Pour chaque context: Diagramme en barres empilées succès, échec par run en fréquence relative
            for context in used_context:                          
                Stack_Diagram(ql_path, action, context, success, failure, "per run")    
                
            return used_success, used_uses #final_uses, final_success
        
def ModuloIndexes(final_feat):
    """
    Compute for each context in final_feat the context number modulo the number of contexts in ecfp level 
    (34 for ecfp 0; 1469 for ecfp 2).
    The goal is to draw results per similar ecfp ondexes dependently the atom added or the type of change bond 

    Args:
        final_feat (pd): Frequency of feature (use or success) for each context
    Returns:
        pd.DataFrame: list of contexts and corresponding sum of use or success 
    """
    #Définition de la valeur du modulo en fonction de l'ecfp considéré
    if ecfp==0: 
        modulo=34
    else: modulo = 4619
    
    mod_indices = {} 
    for key, value in final_feat.items():
        mod_indices[key] = (key % modulo, value)
        
    feat_sum_dict = {}
    for key, value in mod_indices.items():
        mod_index = mod_indices[key][0]
       
        #initialisation of mod_index if new one
        if mod_index not in feat_sum_dict:
            feat_sum_dict[mod_index] = 0
        
        # Sum of use / success with a same mod_index
        feat_sum_dict[mod_index] += mod_indices[key][1]
            # Convert dictionnary to DataFrame
    #print(feat_sum_dict)
    return pd.DataFrame(list(feat_sum_dict.items()), columns=['contexte', 'Sum'])

#####################################
###### Main plottings graphs ########
#####################################
def main(path, config, ql_file, rdm_path, actions_list):
    """
    Plotting graphs
    :param path : path to folder containing results of stochastic Qlearning runs
    :param ql_file : name of the file to consider containing results of stochastic Qlearning runs
    :param rdm_path : path to folder containing results of random runs
    """
    if ql_file == None :
        ql_path = path
    else:
        ql_path = path + ql_file
    #print(ql_path)
    
    use_per_action = {} 
    
#    for action in actions_list:
            #print(action)
#            final_uses, final_success = SuccessRatePlot(ql_path, path, config, action, use_per_action)
                        
            ######################################################################################
            ############ Diagram Success over use per ecfp*(nb_atom U nb_bound) ################
            # Sum of uses for each context modulo 34 pour ecfp0; 4619 pour ecfp2
#            Usage = ModuloIndexes(final_uses) 
            #Delete non used w_ecfp
            #Usage = Usage[Usage['Sum']>0]

            # Sum of succes for each context modulo 34 pour ecfp0; 4619 pour ecfp2    
#            Success = ModuloIndexes(final_success)               
            
            #Concatenate use and sucess data
#            merged = pd.merge(Usage, Success, on='contexte', suffixes=('_use', '_succes'))
            # Failure calculation (use -suces)
#            merged['Sum_echec'] = merged['Sum_use'] - merged['Sum_succes']
#            merged['contexte'] = 'W' + merged['contexte'].astype(str)
            # Création du graphique à barres empilées
#            Stack_Diagram(ql_path, action, merged['contexte'], merged['Sum_succes'], merged['Sum_echec'], "ecfp"+str(ecfp))
    ################ Distribution of use per action #############
    actions = list(use_per_action.keys())
    print(actions)
    nb_use = [values[0] for values in use_per_action.values()]   
    HbarDiagram(ql_path, "Usage_all_action", actions, nb_use, None, "Use count", "Action", "Use count per action type")
    ################ Distribution of contexts counts per action #############   
    nb_contexts = [values[1] for values in use_per_action.values()] 
    HbarDiagram(ql_path, "Context_all_action", actions, nb_contexts, None, "Visited contexts count", "Action", "Visited contexts count per action type")
    
    ############## Mean of weights evolution over steps for each action ##################
    ##GraphOverSteps("Moyennes des success rates par steps", ql_path, config, "_success_rates", ["AddA", "RmA", "ChB"])
    
    
    
##############################################################
#### Appel des méthodes pour chaque ecfp et chaque action ####
##############################################################
#for ecfp in [0]:#[0, 2]: #2ecfp explorés
            #### Plot results EFCP0 ou 2 - 10 run - 500 steps #######
max_steps = 500 #500 #1000
atoms = "C,N,O,F"
epsilon_0 = 1.0
  
for max_depth in {1}:#{1, 2, 3} :
    for epsilon in {0.1}:#0.1, 0.2, 
        for ecfp in {2, 0}:#, 0} :
            for epsilon_method in ["greedy", "power_law", "constant"]:
                if epsilon_method == "greedy":
                    EpsParamList = [0.05, 0.1]#0.1, 0.001, 0.005, 0.01, 0.05, 0.1]
                elif epsilon_method == "power_law":
                    EpsParamList = [0.35]#[0.25, 0.3, 0.35, 0.4]
                else: #epsilon_method == "constant":
                    EpsParamList = [epsilon]

                for EpsParam in EpsParamList:
                    config = f"stoch_ecfp{ecfp}_eps_{epsilon_method}_{EpsParam}_epsmin_{epsilon}_random_alea_ql_steps{str(max_steps)}_depth{str(max_depth)}_{atoms}_sillyTh0"#stoch_ecfp2_eps_power_law_0.3_random_alea_ql_steps500_depth1_C,N,O,F"
                    date = "/"    #"/"
                    Main_path = "./examples/ICTAI/"#Silly_Qlearning/" #qed_Qlearning/"
                    Nb_run = 10 #4#10
                    Spec_path = f"{Nb_run}run_stoch_ecfp{ecfp}_eps_{epsilon_method}_{EpsParam}_epsmin_{epsilon}_random_alea_ql_steps{str(max_steps)}_depth{str(max_depth)}_{atoms}_sillyTh0"

                    ActionList = ["AddA", "RmA", "ChB"]#["addA", "rmA", "chB"]))
    
    # Creating results directory if it doesn't exist
                    os.makedirs(dirname(join(Main_path+Spec_path+date, "file")), exist_ok=True)
                    run_mean_std(Main_path+Spec_path+date, Nb_run, 'steps.csv')
                    rdm_Nb_run = 10
                    rdm_path = "./examples/Silly_Random/10run_Random_steps500_depth1_C,N,O,F_RandomPop_250512/"#10run_Random_steps500_depth3_C,N,O,F_RandomPop_250509/"#10run_Random_steps500_depth1_C,N,O,F_RandomPop_NoSillyThreshold_250509/"#10run_Random_steps500_depth1_C,N,O,F_RandomPop_250512/"#"./examples/Silly_Random/10run_Random_steps500_depth1_C,N,O,F_RandomPop_NoSillyThreshold_250509/"#"./examples/Silly_Random/" 
#10run_Random_steps500_depth3_C,N,O,F_RandomPop_250509/"
#10run_Random_steps500_depth1_C,N,O,F_RandomPop_NoSillyThreshold_250509/"#10run_Random_steps500_depth1_C,N,O,F_RandomPop_250512/"#"./examples/Silly_Random/10run_Random_steps500_depth1_C,N,O,F_RandomPop_NoSillyThreshold_250509/"#"./examples/Silly_Random/" 
                    rdmRunpath = "" #"ImprovMutStrat_steps500_depth1_C,N,O,F"
                    run_mean_std(rdm_path, rdm_Nb_run, 'steps.csv')
                    qlpath = Main_path+Spec_path+date
                    SumAllRun(qlpath, config, Nb_run, ActionList)

                    main(qlpath, Spec_path, None, rdm_path, ActionList)
    #uniquement pour la globalité des Nb_run run, moyenne et médiane des tabu fails, silly walk et timestamps
                    for stat in ["mean", "median"]:
                        plotfail(qlpath, stat+"_steps.csv", rdm_path + stat+"_steps.csv", config, ecfp, stat,100)
    #plotfail(qlpath, stat+"_steps.csv", rdm_path + "ImprovMutStrat_steps500_depth1_C,N,O,F_"+stat+"_steps.csv", config, ecfp, stat,100)
        
                    for i in range(1, max(Nb_run,rdm_Nb_run)+1):  #10 run
                        if __name__ == "__main__":
        ####main(qlpath, config, config+"_run"+str(i), rdm_path+rdmRunpath+"_run"+str(i), ActionList)
                            if i <= min(Nb_run,rdm_Nb_run)+1 :
 #           plotfail(qlpath+config+"_run"+str(i)+"/", "steps.csv", rdm_path +rdmRunpath+"_run"+str(i)+"/steps.csv", 
 #                    config, ecfp, "",100)
                                plotfail(qlpath+"run"+str(i)+"/", "steps.csv", rdm_path +"run"+str(i)+"/steps.csv", 
                                            config, ecfp, "",100)            
 