import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pandas import read_csv
import seaborn as sns
import plotly.express as px

##########################################################
################### Plot over steps ######################
##########################################################
def PlotOverSteps(ql0_steps_reader, ql2_steps_reader, rdm_steps_reader, data, title, xlabel, ylabel, stat, div_nbstep):
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
    ql0_data = np.array(ql0_steps_reader[data].tolist()) 
    ql2_data = np.array(ql2_steps_reader[data].tolist()) 
    rdm_data = np.array(rdm_steps_reader[data].tolist())
    
    # Execution time evolution over steps
    plt.figure("Evolution of realism percent over steps", figsize=(6, 4))
   # plt.title("Evolution of realism percent over steps")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    print(len(ql0_data))
    print(len(ql2_data))
    print(len(rdm_data))
    print(int(np.floor(len(ql0_data) / div_nbstep)))

    ql0_mean = [ql0_data[i : i + int(np.floor(len(ql0_data) / div_nbstep))].mean() for i in range(0, len(ql0_data), int(np.floor(len(ql0_data) / div_nbstep)))]
    ql2_mean = [ql2_data[i : i + int(np.floor(len(ql2_data) / div_nbstep))].mean() for i in range(0, len(ql2_data), int(np.floor(len(ql2_data) / div_nbstep)))]
    rdm_mean = [rdm_data[i : i + int(np.floor(len(rdm_data) / div_nbstep))].mean() for i in range(0, len(rdm_data), int(np.floor(len(rdm_data) / div_nbstep)))]
    plt.plot(np.array(range(0, len(rdm_data), int(np.floor(len(rdm_data) / div_nbstep)))), rdm_mean, color='firebrick', label="EvoMol")
    plt.plot(np.array(range(0, len(ql0_data), int(np.floor(len(ql0_data) / div_nbstep)))), ql0_mean, color='dodgerblue', label="ECFP0")
    plt.plot(np.array(range(0, len(ql2_data), int(np.floor(len(ql2_data) / div_nbstep)))), ql2_mean, color='forestgreen', label="ECFP2")

    rdm_std = [rdm_data[i : i + int(np.floor(len(rdm_data) / div_nbstep))].std() for i in range(0, len(rdm_data), int(np.floor(len(rdm_data) / div_nbstep)))]
    ql0_std = [ql0_data[i : i + int(np.floor(len(ql0_data) / div_nbstep))].std() for i in range(0, len(ql0_data), int(np.floor(len(ql0_data) / div_nbstep)))]
    ql2_std = [ql2_data[i : i + int(np.floor(len(ql2_data) / div_nbstep))].std() for i in range(0, len(ql2_data), int(np.floor(len(ql2_data) / div_nbstep)))]
    
    ql0_max = [x + y for x, y in zip(ql0_mean, ql0_std)]
    ql0_min = [x - y for x, y in zip(ql0_mean, ql0_std)]
    ql2_max = [x + y for x, y in zip(ql2_mean, ql2_std)]
    ql2_min = [x - y for x, y in zip(ql2_mean, ql2_std)]
    rdm_max = [x + y for x, y in zip(rdm_mean, rdm_std)]
    rdm_min = [x - y for x, y in zip(rdm_mean, rdm_std)]

    #plt.plot(np.array(range(0, len(ql0_data), int(np.floor(len(ql0_data) / div_nbstep)))), ql0_max, color='dodgerblue', linestyle='--', label="mean + std ECFP0")
    #plt.plot(np.array(range(0, len(ql2_data), int(np.floor(len(ql2_data) / div_nbstep)))), ql2_max, color='forestgreen', linestyle='--', label="mean + std ECFP2")
    #plt.plot(np.array(range(0, len(rdm_data), int(np.floor(len(rdm_data) / div_nbstep)))), rdm_max, color='firebrick', linestyle='--', label="mean + std EvoMol")
    #plt.plot(np.array(range(0, len(ql0_data), int(np.floor(len(ql0_data) / div_nbstep)))), ql0_min, color='dodgerblue', linestyle='-.', label="mean - std ECFP0")
    #plt.plot(np.array(range(0, len(ql2_data), int(np.floor(len(ql2_data) / div_nbstep)))), ql2_min, color='forestgreen', linestyle='-.', label="mean - std ECFP2")
    #plt.plot(np.array(range(0, len(rdm_data), int(np.floor(len(rdm_data) / div_nbstep)))), rdm_min, color='firebrick', linestyle='-.', label="mean - std EvoMol")


    plt.fill_between(np.array(range(0, len(rdm_data), int(np.floor(len(rdm_data) / div_nbstep)))), rdm_max , rdm_min, color='firebrick', alpha=.1)
    plt.fill_between(np.array(range(0, len(ql0_data), int(np.floor(len(ql0_data) / div_nbstep)))), ql0_max , ql0_min, color='dodgerblue', alpha=.1)
    plt.fill_between(np.array(range(0, len(ql2_data), int(np.floor(len(ql2_data) / div_nbstep)))), ql2_max , ql2_min, color='forestgreen', alpha=.1)

    ql0_total_mean = np.array(ql0_data[1:]).mean()
    ql2_total_mean = np.array(ql2_data[1:]).mean()
    rdm_total_mean = np.array(rdm_data[1:]).mean()

    plt.plot(np.array(range(0, len(rdm_data), int(np.floor(len(rdm_data) / div_nbstep)))), [rdm_total_mean] * len(rdm_mean), color='firebrick', linestyle=':', label=stat+" EvoMol")
    plt.plot(np.array(range(0, len(ql0_data), int(np.floor(len(ql0_data) / div_nbstep)))), [ql0_total_mean] * len(ql0_mean), color='dodgerblue', linestyle=':', label=stat+" ECFP0")
    plt.plot(np.array(range(0, len(ql2_data), int(np.floor(len(ql2_data) / div_nbstep)))), [ql2_total_mean] * len(ql2_mean), color='forestgreen', linestyle=':', label=stat+" ECFP2 ")
    
    plt.annotate(np.round(rdm_total_mean,2), xy=(1, rdm_total_mean), xytext=(8, 0),
                    xycoords=('axes fraction', 'data'), textcoords='offset points')#, fontsize=16)
    plt.annotate(np.round(ql0_total_mean,2), xy=(1, ql0_total_mean), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')#, fontsize=16)
    plt.annotate(np.round(ql2_total_mean,2), xy=(1, ql2_total_mean), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')#, fontsize=16)
    
    plt.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.75,
            alpha=0.2)

    plt.xlim([0, len(ql0_data)])
    
    plt.ylim([0, 1])
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("./examples/ICTAI_graphs/" + "/"+data+"_" + str(int(np.floor(len(ql0_data) / div_nbstep))) + "steps.png")
    plt.close()

def boxPlot(ql0_steps_reader, ql2_steps_reader, rdm_steps_reader, feat, eps, ylabel, stat):
    """
    Boxplot over step (used for timestamp, tabu list fail or silly walk failed)
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
    """
    ql0_data = np.array(ql0_steps_reader[feat].tolist()) 
    ql2_data = np.array(ql2_steps_reader[feat].tolist()) 
    rdm_data = np.array(rdm_steps_reader[feat].tolist())

    data = {
    'Group': ['EvoMol']*500 + ['ECFP0']*500 + ['ECFP2']*500,
    'Value': np.concatenate([rdm_data, ql0_data, ql2_data])
}

    df = pd.DataFrame(data)
# Create additional grouping data
   # df['Subgroup'] = np.random.choice(['X', 'Y'], size=1500)
   # print(df.head())
# Plots graph
    plt.figure("Boxplot over steps", figsize=(6, 4))
    #sns.boxplot(x='Group', y='Value', data=df, hue='Subgroup', palette=["dodgerblue", "forestgreen", "firebrick"])
    sns.boxplot(x='Group', y='Value', data=df, palette=["firebrick", "dodgerblue", "forestgreen"])
    
    #plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.xticks([0, 1, 2], ["EvoMol", "ECFP0", "ECFP2"])
    
    plt.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.75,
            alpha=0.2)

    plt.tight_layout()
    
    plt.savefig("./examples/ICTAI_graphs/" + "/"+feat+"_eps"+str(eps)+"_boxPlot" + str(stat) + ".png")
    
    plt.close()


def SubgroupboxPlot(rdm_steps_reader, ql0_steps_reader_eps1, ql2_steps_reader_eps1, ql0_steps_reader_eps2, ql2_steps_reader_eps2, ql0_steps_reader_eps3, ql2_steps_reader_eps3, feat, ylabel, stat):
    """
    Boxplot over step (used for timestamp, tabu list fail or silly walk failed)
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
    """
    rdm_data = np.array(rdm_steps_reader[feat].tolist())
    ql0_data_eps1 = np.array(ql0_steps_reader_eps1[feat].tolist()) 
    ql2_data_eps1 = np.array(ql2_steps_reader_eps1[feat].tolist()) 
    ql0_data_eps2 = np.array(ql0_steps_reader_eps2[feat].tolist()) 
    ql2_data_eps2 = np.array(ql2_steps_reader_eps2[feat].tolist()) 
    ql0_data_eps3 = np.array(ql0_steps_reader_eps3[feat].tolist()) 
    ql2_data_eps3 = np.array(ql2_steps_reader_eps3[feat].tolist()) 

    data = {
    'Group': ['EvoMol']*500 + ['eps 0.1']*1000 + ['eps 0.2']*1000 + ['eps 0.3']*1000,
    'Subgroup': ['EvoMol']*500 + ['ECFP0']*500 + ['ECFP2']*500 + ['ECFP0']*500 + ['ECFP2']*500 + ['ECFP0']*500 + ['ECFP2']*500,
    'Value': np.concatenate([rdm_data, ql0_data_eps1, ql2_data_eps1, ql0_data_eps2, ql2_data_eps2, ql0_data_eps3, ql2_data_eps3])
}

    df = pd.DataFrame(data)
# Plots graph
    plt.figure("Boxplot over steps", figsize=(6, 4))
    sns.boxplot(x='Group', y='Value', data=df, hue='Subgroup', palette=["firebrick", "dodgerblue", "forestgreen"])
    
    #plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.xticks([0, 1, 2, 3], ["EvoMol", "epsilon 0.1", "epsilon 0.2", "epsilon 0.3"])
    
    plt.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.75,
            alpha=0.2)

    plt.tight_layout()
    
    plt.savefig("./examples/ICTAI_graphs/" + "/"+feat+"_Subgroup_by_eps_boxPlot" + str(stat) + ".png")
    
    plt.close()


def SubgroupViolinPlot(rdm_steps_reader, ql0_steps_reader_eps1, ql2_steps_reader_eps1, ql0_steps_reader_eps2, ql2_steps_reader_eps2, ql0_steps_reader_eps3, ql2_steps_reader_eps3, feat, ylabel, stat):
    """
    Boxplot over step (used for timestamp, tabu list fail or silly walk failed)
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
    """
    rdm_data = np.array(rdm_steps_reader[feat].tolist())
    ql0_data_eps1 = np.array(ql0_steps_reader_eps1[feat].tolist()) 
    ql2_data_eps1 = np.array(ql2_steps_reader_eps1[feat].tolist()) 
    ql0_data_eps2 = np.array(ql0_steps_reader_eps2[feat].tolist()) 
    ql2_data_eps2 = np.array(ql2_steps_reader_eps2[feat].tolist()) 
    ql0_data_eps3 = np.array(ql0_steps_reader_eps3[feat].tolist()) 
    ql2_data_eps3 = np.array(ql2_steps_reader_eps3[feat].tolist()) 

    data = {
    'Group': ['EvoMol']*500 + ['eps 0.1']*1000 + ['eps 0.2']*1000 + ['eps 0.3']*1000,
    'Subgroup': ['EvoMol']*500 + ['ECFP0']*500 + ['ECFP2']*500 + ['ECFP0']*500 + ['ECFP2']*500 + ['ECFP0']*500 + ['ECFP2']*500,
    'Value': np.concatenate([rdm_data, ql0_data_eps1, ql2_data_eps1, ql0_data_eps2, ql2_data_eps2, ql0_data_eps3, ql2_data_eps3])
}

    df = pd.DataFrame(data)
# Plots graph
    plt.figure("violinplot over steps", figsize=(6, 4))
    sns.violinplot(x='Group', y='Value', data=df, hue='Subgroup', palette=["firebrick", "dodgerblue", "forestgreen"])
    
    #plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.xticks([0, 1, 2, 3], ["EvoMol", "epsilon 0.1", "epsilon 0.2", "epsilon 0.3"])
    
    plt.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.75,
            alpha=0.2)

    plt.tight_layout()
    
    plt.savefig("./examples/ICTAI_graphs/" + "/"+feat+"_Subgroup_by_eps_ViolinPlot" + str(stat) + ".png")
    
    plt.close()



def ImportData(path, method, eps):
    steps_reader = pd.read_csv(path+"/mean_steps.csv")
    
    #######################  Deletion of line Step 0   #################################""
    steps_reader.drop(steps_reader[steps_reader['n_replaced'] == 0.0 ].index, inplace=True )#index[[0]])
    
    steps_reader["pct_realism"] = 1 - (steps_reader['n_discarded_sillywalks'] / (steps_reader['n_replaced']+steps_reader['n_discarded_sillywalks']+steps_reader['n_discarded_tabu']))
        
    steps_reader.to_csv("./examples/ICTAI_graphs/"+method+"_"+str(eps)+".csv", decimal=".", index=True)
    return steps_reader

######################################################################################
##################### Plotting tabu list fail and silly walk failed ######################
######################################################################################
def plotfail(stat, div_nbstep):
    """
    Plotting tabu list fail and silly walk failed
    :param ql_path : path to folder containing results of stochastic Qlearning runs
    :param ql_file : name of the file to consider containing results of stochastic Qlearning runs
    :param rdm_file : name of the file to consider containing results of random runs
    :param div_nbstep : diviser used to limit the nb of steps considered as one point in graphs
    :param efcp : ecfp even number considered for stochastic Qlearning runs
    :param stat : stat to perform (mean, median)
    """
    ql0_steps_reader_eps1 = ImportData("./examples/ICTAI_graphs/10run_stoch_ecfp0_eps_power_law_0.35_epsmin_0.1_random_alea_ql_steps500_depth1_C,N,O,F_sillyTh0",
               "ecfp0", 0.1)
    ql2_steps_reader_eps1 = ImportData("./examples/ICTAI_graphs/10run_stoch_ecfp2_eps_power_law_0.35_epsmin_0.1_random_alea_ql_steps500_depth1_C,N,O,F_sillyTh0",
                "ecfp2", 0.1)
    rdm_steps_reader = ImportData("./examples/Silly_Random/10run_Random_steps500_depth1_C,N,O,F_RandomPop_250512",
                "EvoMol", 0.1)    
    
    PlotOverSteps(ql0_steps_reader_eps1, ql2_steps_reader_eps1, rdm_steps_reader,'n_discarded_sillywalks',stat+"s Nombre de molécules ne passant pas le filtre des silly walks ","Steps" ,"Failure frequency", stat, div_nbstep)
    PlotOverSteps(ql0_steps_reader_eps1, ql2_steps_reader_eps1, rdm_steps_reader,'pct_realism',stat+"s realism percent ","Steps" ,"Realism frequency", stat, div_nbstep)                
    boxPlot(ql0_steps_reader_eps1, ql2_steps_reader_eps1, rdm_steps_reader,"pct_realism",0.1,"Realism frequency", stat)
    ql0_steps_reader_eps2 = ImportData("./examples/ICTAI_graphs/10run_stoch_ecfp0_eps_power_law_0.35_epsmin_0.2_random_alea_ql_steps500_depth1_C,N,O,F_sillyTh0",
               "ecfp0", 0.2)
    ql2_steps_reader_eps2 = ImportData("./examples/ICTAI_graphs/10run_stoch_ecfp2_eps_power_law_0.35_epsmin_0.2_random_alea_ql_steps500_depth1_C,N,O,F_sillyTh0",
                "ecfp2", 0.2) 
    boxPlot(ql0_steps_reader_eps2, ql2_steps_reader_eps2, rdm_steps_reader, "pct_realism", 0.2 ,"Realism frequency", stat)
    
    ql0_steps_reader_eps3 = ImportData("./examples/ICTAI_graphs/10run_stoch_ecfp0_eps_power_law_0.35_epsmin_0.3_random_alea_ql_steps500_depth1_C,N,O,F_sillyTh0",
               "ecfp0", 0.3)
    ql2_steps_reader_eps3 = ImportData("./examples/ICTAI_graphs/10run_stoch_ecfp2_eps_power_law_0.35_epsmin_0.3_random_alea_ql_steps500_depth1_C,N,O,F_sillyTh0",
                "ecfp2", 0.3)
    boxPlot(ql0_steps_reader_eps3, ql2_steps_reader_eps3, rdm_steps_reader, "pct_realism", 0.3 ,"Realism frequency", stat)

    SubgroupboxPlot(rdm_steps_reader, ql0_steps_reader_eps1, ql2_steps_reader_eps1, ql0_steps_reader_eps2, ql2_steps_reader_eps2, ql0_steps_reader_eps3, ql2_steps_reader_eps3, "pct_realism", "Realism frequency", stat)
    SubgroupViolinPlot(rdm_steps_reader, ql0_steps_reader_eps1, ql2_steps_reader_eps1, ql0_steps_reader_eps2, ql2_steps_reader_eps2, ql0_steps_reader_eps3, ql2_steps_reader_eps3, "pct_realism", "Realism frequency", stat)

    
plotfail("mean", 25)      
plotfail("mean", 10) 
plotfail("mean", 50) 