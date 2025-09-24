########################################################################################
# 2. Stat_interRun_plot_pct_succ.py                                                    # 
########################################################################################
# Script to compare Evomol and Evomol-RL algorithms across different configurations    #
########################################################################################
# - For each configuration:                                                            #
#   . Calculates inter run ratios and complementary percentages for discarded features #  
#   . Generates statistical summaries and visualizations                               #
#   . Outputs results in structured JSON and CSV formats                               #
#                                                                                      #   
# - Select the best configuration over multiple runs based on realism percentage       #    
# maximum mean                                                                         #
# - Run Kruskal-Wallis test to compare realism percentages between Evomol and best     #
# EvoMol-RL configuration                                                              #
#                                                                                      #
# Author : Gaëlle Milon-Harnois                                                        #
# Date : June 2025                                                                     #
########################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple
import json
import csv

class AlgorithmAnalyzer:
    """
    Analyzer for comparing Evomol and Evomol-RL algorithms across different configurations.
    Calculates ratios and complementary percentages for discarded features.
    """
    def __init__(self, methods_config: Dict[str, List[float]], ecfp_values: List[int] = None, eps_values: List[float] = None, 
                 n_runs: int = 10, output_dir: str = "."):
        self.eps_values = eps_values or [0.1, 0.2, 0.3]
        self.ecfp_values = ecfp_values or [0, 2]
        self.n_runs = n_runs
        self.output_dir = output_dir
        self.methods_config = methods_config

        # Features to calculate and report
        self.target_features = [
            'Pct_discarded_sillywalks',
            'Pct_discarded_tabu',
            'Pct_realism',
            'Pct_novelty'
        ]
         # Storage for results
        self.results = {}
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
    def calculate_ratios(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate ratios for the last row of a dataframe.
        
        Args:
            df: DataFrame containing the experimental data
            
        Returns:
            Dictionary containing calculated ratios and percentages
        """
        try:
            # Get the last row of the dataframe
            last_Step = df.iloc[-1] #last line of the dataframe contains the cumulative totals
        
            # Compute the denominator (sum of the three total columns)
            denominator = (last_Step['TOTAL_discarded_sillywalks'] + 
                       last_Step['TOTAL_discarded_tabu'] + 
                       last_Step['TOTAL_objective_calls'])
    
      # Handle division by zero
            if denominator == 0:
                print("Warning: Denominator is zero, returning default values")
                return {
                    'Pct_discarded_sillywalks': None,
                    'Pct_discarded_tabu': None,
                    'Pct_realism': None,
                    'Pct_novelty': None
                        }
            
            # Calculate ratios
            ratio_sillywalks = last_Step['TOTAL_discarded_sillywalks'] / denominator
            ratio_tabu = last_Step['TOTAL_discarded_tabu']  / denominator
            
            # Calculate complementary percentages
            pct_realism = 1 - ratio_sillywalks
            pct_novelty = 1 - ratio_tabu
            
            return {
                'Pct_discarded_sillywalks': ratio_sillywalks,
                'Pct_discarded_tabu': ratio_tabu,
                'Pct_realism': pct_realism,
                'Pct_novelty': pct_novelty
            }
        except KeyError as e:
            raise KeyError(f"Missing required column in dataframe: {e}")
        except Exception as e:
            raise Exception(f"Error calculating ratios: {e}")        
    
    def load_dataframe(self, file_path: str) -> pd.DataFrame:
        """
        Load dataframe for a specific configuration and run.
        Args:
            file_path: Path to the CSV file
        Returns:
            DataFrame with the required columns
        """
        try:
            df = pd.read_csv(file_path)  
            return df
        except FileNotFoundError:
            print(f"File {file_path} not found.")
        except Exception as e:
            print(f"An error occurred while loading {file_path}: {e}")

    #Save interrun statistics per steps to draw the global %realism graph
    def save_stat_all_run(self, stat, headers, file_name):
        """saving generated statistics to csv file
        Args:
            stat (_type_): _description_
            headers (_type_): _description_
            file_name (text): path and csv file name
        """
        Dstat_df = pd.DataFrame(stat, columns=headers)
        Dstat_df.to_csv(file_name, index=False)


    def analyze_and_save_statistics(self, stacked_data, headers, path, filename, 
                               stats=['sum', 'mean', 'std', 'median'], 
                               save_function=None):
        """
        Performs statistical analysis on stacked data and saves the results.
        Parameters:
        -----------
        stacked_data : numpy.ndarray
            Stacked data with runs dimension on axis 2
        headers : list
            Column headers for the data
        path : str
            Output file path
        filename : str
            Base filename for output files
        stats : list, optional
            List of statistics to calculate ['sum', 'mean', 'std', 'median']
        save_function : callable, optional
            Custom save function. If None, uses save_stat_all_run
    
        Returns:
        --------
        dict : Dictionary containing computed statistical results
        """
        # Dictionary of available statistical functions
        stat_functions = {
            'sum': np.nansum,
            'mean': np.nanmean, 
            'std': np.nanstd,
            'median': np.nanmedian,
            'min': np.nanmin,
            'max': np.nanmax,
            'var': np.nanvar,
            'percentile_25': lambda data, axis: np.nanpercentile(data, 25, axis=axis),
            'percentile_75': lambda data, axis: np.nanpercentile(data, 75, axis=axis)
        }
        # Use save_stat_all_run by default if no function is provided
        if save_function is None:
            save_function = self.save_stat_all_run
        # Calculate and save each requested statistic
        for stat in stats:
            if stat in stat_functions:
                # Calculate the statistic
                result = stat_functions[stat](stacked_data, axis=2)
                output_filename = f"{path}{stat}_{filename}"
                save_function(result, headers, output_filename)
            else:
                print(f"⚠️ Statistic '{stat}' not recognized. Available: {list(stat_functions.keys())}")    

    def StackInterRunData(self, all_data):
        """Compute and save inter-run statistics per steps

        Args:
            all_data (list): list of dataframes for each run
            headers (list): list of column headers
            file_path (text): path to save the statistics files
            stats (list): list of statistics to compute (mean, std, median, sum)
            save_function (function): function to save the computed statistics
        Returns:
        --------
        dict : Dictionnaire contenant les résultats des statistiques calculées
        """
        j=1
        for df in all_data:
            j+=1
        # Stack all DataFrames in one to compute statistics
        return np.stack([df.values for df in all_data], axis=2)

    def run_evomol_analysis(self, file_path: str, precision) -> List[Dict[str, float]]:
        """
        Run analysis for Evomol algorithm across multiple runs.
        
        Args:
            data_generator_func: Function that generates a DataFrame for each run
            
        Returns:
            List of ratio calculations for each run
        """
        # Initialising lists
        all_data = [] # to keep nb_run files data
        run_results = [] # to keep results of each run
        
        for run_id in range(self.n_runs):
            # Generate dataframe for this run
            df = self.load_dataframe(f"{file_path}{precision}/run{run_id+1}/steps.csv")  

            # Headers keeping for graph purpose:
            if run_id==1: 
                headers = pd.read_csv(f"{file_path}{precision}/run{run_id+1}/steps.csv").columns
            # Convert data from string to float
            df = df.apply(pd.to_numeric, errors='coerce')
            all_data.append(df)

            # Calculate ratios
            ratios = self.calculate_ratios(df)
            ratios['run_id'] = run_id+1
            run_results.append(ratios)
        
        # Save individual run results
        self.save_results_to_file(run_results, f"./examples/Evomol_results{precision}.json")
        
        # Stack all runs data for inter-run statistics calculation
        stacked_data = np.stack([df.values for df in all_data], axis=2)
        self.analyze_and_save_statistics(stacked_data, headers, f"{file_path}{precision}/", "steps.csv",
                                         stats=['sum', 'mean', 'std', 'median'], 
                                         save_function=None)
        return run_results
    
    def run_evomol_rl_analysis(self, file_path, precision) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
        """
        Run analysis for Evomol-RL algorithm with different epsilon values and methods.
        
        Args:
            data_generator_func: Function that generates DataFrame given run_id, eps, method, param
            eps_values: List of epsilon values to test
            methods_config: Dictionary mapping method names to parameter lists
            
        Returns:
            Nested dictionary with results organized by eps, method, and parameter
        """
        # Initialising lists
        rl_results = {}
                       
        for eps in self.eps_values:
            rl_results[f"eps_{eps}"] = {}
            for method, param_list in self.methods_config.items():
                rl_results[f"eps_{eps}"][method]= {}
                for param in param_list:
                    if method == "constant":
                        param = eps
                    rl_results[f"eps_{eps}"][method][f"{method}_param_{param}"] = {}
                    for ecfp in self.ecfp_values:
                        all_data = [] # to keep nb_run files data   
                        run_results = []
                        Complete_File_Path = f"{file_path}10run_stoch_ecfp{ecfp}_eps_{method}_{param}_epsmin_{eps}_random_alea_ql_steps500_depth1_C,N,O,F{precision}/"
                        for run_id in range(self.n_runs):
                        # Generate dataframe for this configuration and run
                            df = self.load_dataframe(f"{Complete_File_Path}run{run_id+1}/steps.csv")  

                            # Headers keeping for graph purpose:
                            if run_id==1: 
                                headers = pd.read_csv(f"{Complete_File_Path}run{run_id+1}/steps.csv").columns
                            # Convert data from string to float
                            df = df.apply(pd.to_numeric, errors='coerce')
                            all_data.append(df)

                        # Calculate ratios
                            ratios = self.calculate_ratios(df)
                            ratios['run_id'] = run_id+1
                            run_results.append(ratios)
                        
                        #Save results of each run
                        rl_results[f"eps_{eps}"][method][f"{method}_param_{param}"][f"ecfp{ecfp}"] = run_results

                        # Stack all runs data for inter-run statistics calculation
                        stacked_data = np.stack([df.values for df in all_data], axis=2)
                        # Inter-run statistics calculation
                        self.analyze_and_save_statistics(stacked_data, headers, Complete_File_Path, "steps.csv",
                                                         stats=['sum', 'mean', 'std', 'median'], 
                                                         save_function=None)
        self.save_results_to_file(rl_results, f"./examples/Evomol_RL_results{precision}.json")
        return rl_results
    
    def calculate_statistics(self, run_results: List[Dict[str, float]], max_realism, max_path, config) -> Dict[str, Dict[str, float]]:
        """
        Calculate mean and standard deviation for each metric across runs.
        
        Args:
            run_results: List of dictionaries containing ratios for each run
            
        Returns:
            Dictionary with mean and std for each metric
        """
        metrics = ['Pct_discarded_sillywalks', 'Pct_discarded_tabu', 
                  'Pct_realism', 'Pct_novelty']
        
        stats = {}
        for metric in metrics:
            values = [result[metric] for result in run_results]
            stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1)  # Sample standard deviation
            }
            if np.mean(values) > max_realism and metric == 'Pct_realism':
                max_realism = np.mean(values)
                max_path = [config]
            elif np.mean(values) == max_realism:
                max_path = max_path + config
        return stats, max_realism, max_path
    
    def save_results_to_file(self, results: Dict, filename: str):
        """
        Save results to a JSON file.
        
        Args:
            results: Dictionary containing all results
            filename: Name of the output file
        """
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Deep convert the results dictionary
        def deep_convert(d):
            if isinstance(d, dict):
                return {k: deep_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [deep_convert(item) for item in d]
            else:
                return convert_numpy(d)
        
        converted_results = deep_convert(results)
        
        with open(filename, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def create_comparison_table(self, evomol_stats: Dict, evomol_rl_stats: Dict, 
                               target_config: Tuple[str, float, float], metrics) -> pd.DataFrame:
        """
        Create the final comparison table as specified.
        
        Args:
            evomol_stats: Statistics for Evomol algorithm
            evomol_rl_stats: Statistics for Evomol-RL algorithm
            target_config: Tuple of (method, param, eps) for the specific configuration
            
        Returns:
            DataFrame with the comparison table
        """
        method, param, eps_target = target_config
        eps_values = [0.1, 0.2, 0.3]
        ecfpCl = ['ecfp0', 'ecfp2']  # As mentioned in requirements
        
        # Create multi-index columns
        columns = pd.MultiIndex.from_product([metrics, ecfpCl], names=['', 'EvoMol-RL'])
        
        # Create index for epsilon values
        index = pd.Index(eps_values, name='Epsilon')
        
        # Initialize DataFrame
        comparison_df = pd.DataFrame(index=index, columns=columns)
        
        # Fill the table
        # Evomol values (same for all eps since Evomol doesn't use eps)
        for metric in metrics:
            evomol_mean = evomol_stats[metric]['mean']  # Example metric
            evomol_std = evomol_stats[metric]['std']
            comparison_df.loc['Evomol', metric] = f"{evomol_mean:.4f} ± {evomol_std:.4f}"
                
        for ecfp in self.ecfp_values:
            for eps in self.eps_values:
                for metric in metrics:
                # Evomol-RL values for the target configuration
                    rl_mean = evomol_rl_stats[f"eps_{eps}"][method][f"{method}_param_{param}"][f"ecfp{ecfp}"][metric]['mean']
                    rl_std = evomol_rl_stats[f"eps_{eps}"][method][f"{method}_param_{param}"][f"ecfp{ecfp}"][metric]['std']
                    comparison_df.loc[eps, (metric, f"ecfp{ecfp}")] = f"{rl_mean:.4f} ± {rl_std:.4f}"
        return comparison_df
    
    ################### Plot over steps ######################
    def PlotOverSteps(self, ql0_steps_reader, ql2_steps_reader, rdm_steps_reader, 
                      data, title, xlabel, ylabel, stat, step_nb, config):
        """
        Plotting over step (used for timestamp, tabu list fail (or novelty percent) or silly walk failed (or realism percent))
        :param ql0_steps_reader : path and file to consider for stochastic Qlearning runs with ecfp0
        :param ql2_steps_reader : path and file to consider for stochastic Qlearning runs with ecfp2
        :param rdm_steps_reader : path and file to consider for random runs
        :param data : feature to plot ( 'timestamps', 'n_discarded_tabu', 'n_discarded_sillywalks', ...)
        :param title : graph title
        :param xlabel : label of horizontal axis 
        :param ylabel : label of vertical axis
        :param stat : stat to perform (mean+-std, median-IQR)
        :param step_nb : number of steps to consider as one point in graphs calculated on sliding window
        """
        ql0_data = np.array(ql0_steps_reader[data].tolist()) 
        ql2_data = np.array(ql2_steps_reader[data].tolist()) 
        rdm_data = np.array(rdm_steps_reader[data].tolist())
    
        # Execution time evolution over steps
        plt.figure(title, figsize=(6, 4))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        x_min = 1

        # Function to compute means on SW until the last step
        def calculate_sliding_stats(data, step_nb, x_min):
            means = []
            stds = []
            indices = []
        
            i = x_min
            #Complete window management
            while i + step_nb < len(data)+1:
                window_data = data[i:i + step_nb]
                means.append(window_data.mean())
                stds.append(window_data.std())
                indices.append(i+ step_nb)
                i += step_nb

            if i<len(data):
                window_data = data[i:]
                means.append(window_data.mean())
                stds.append(window_data.std())
                indices.append(i+ step_nb if i + step_nb < len(data) else len(data))

            return means, stds, indices

        # Compute statistics for each dataset
        rdm_mean, rdm_std, rdm_indices = calculate_sliding_stats(rdm_data, step_nb, x_min)
        ql0_mean, ql0_std, ql0_indices = calculate_sliding_stats(ql0_data, step_nb, x_min)
        ql2_mean, ql2_std, ql2_indices = calculate_sliding_stats(ql2_data, step_nb, x_min)

        #means plot
        plt.plot(rdm_indices, rdm_mean, color='firebrick', label="EvoMol")
        plt.plot(ql0_indices, ql0_mean, color='dodgerblue', label="ECFP0")
        plt.plot(ql2_indices, ql2_mean, color='forestgreen', label="ECFP2")

        # Compute boubded for error bands
        ql0_max = [x + y for x, y in zip(ql0_mean, ql0_std)]
        ql0_min = [x - y for x, y in zip(ql0_mean, ql0_std)]
        ql2_max = [x + y for x, y in zip(ql2_mean, ql2_std)]
        ql2_min = [x - y for x, y in zip(ql2_mean, ql2_std)]
        rdm_max = [x + y for x, y in zip(rdm_mean, rdm_std)]
        rdm_min = [x - y for x, y in zip(rdm_mean, rdm_std)]

        # Plot error bands in slightly transparent colors
        plt.fill_between(rdm_indices, rdm_max, rdm_min, color='firebrick', alpha=.1)
        plt.fill_between(ql0_indices, ql0_max, ql0_min, color='dodgerblue', alpha=.1)
        plt.fill_between(ql2_indices, ql2_max, ql2_min, color='forestgreen', alpha=.1)
        
        # Compute global means
        rdm_total_mean = np.array(rdm_data[1:]).mean()
        rdm_total_std = np.array(rdm_data[1:]).std()
        ql0_total_mean = np.array(ql0_data[1:]).mean()
        ql0_total_std = np.array(ql0_data[1:]).std()
        ql2_total_mean = np.array(ql2_data[1:]).mean()
        ql2_total_std = np.array(ql2_data[1:]).std()

    # Drawing global means (dashed lines)
        # Indices management to extend lines to the end of the graph
        full_rdm_x = rdm_indices + [len(rdm_data)] if rdm_indices[-1] < len(rdm_data) else rdm_indices
        full_ql0_x = ql0_indices + [len(ql0_data)] if ql0_indices[-1] < len(ql0_data) else ql0_indices
        full_ql2_x = ql2_indices + [len(ql2_data)] if ql2_indices[-1] < len(ql2_data) else ql2_indices
    
        plt.plot(full_rdm_x, [rdm_total_mean] * len(full_rdm_x), color='firebrick', linestyle=':', label=stat+" EvoMol")
        plt.plot(full_ql0_x, [ql0_total_mean] * len(full_ql0_x), color='dodgerblue', linestyle=':', label=stat+" ECFP0")
        plt.plot(full_ql2_x, [ql2_total_mean] * len(full_ql2_x), color='forestgreen', linestyle=':', label=stat+" ECFP2 ")

        plt.annotate(np.round(rdm_total_mean,2), xy=(1, rdm_total_mean-0.02), xytext=(8, 0),
                    xycoords=('axes fraction', 'data'), textcoords='offset points')
        plt.annotate(np.round(ql0_total_mean,2), xy=(1, ql0_total_mean-0.02), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
        plt.annotate(np.round(ql2_total_mean,2), xy=(1, ql2_total_mean), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
    
        plt.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.75,
            alpha=0.2)

        plt.xlim([step_nb, len(ql0_data)])
    
        plt.ylim([0, 1])
    
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir + "/"+ylabel+"_"+config+"_"+ str(step_nb) + f"steps_{stat}.svg", format='svg', dpi=1200)
        plt.close()

################### Import data and compute realism percentage ######################
def ImportData(path, stat, method, eps):
    """Import data from csv file and compute realism percentage
    Args:
        path (text): path to folder containing results
        stat (text): stat to consider (mean, median, sum)
        method (text): method used (greedy, power_law, constant)
        eps (float): epsilon value used (0.1, 0.2, 0.3)
    Returns:
        steps_reader (dataframe): dataframe containing the data with realism percentage added
    """
    steps_reader = pd.read_csv(path+"/"+stat+"_steps.csv")
    steps_reader["pct_realism"] = 1 - (steps_reader['n_discarded_sillywalks'] / (steps_reader['n_replaced']+steps_reader['n_discarded_sillywalks']+steps_reader['n_discarded_tabu']))
    steps_reader.to_csv("./examples/"+stat+"_"+method+"_"+str(eps)+".csv", decimal=".", index=True)
    return steps_reader

def main(target_eps = 0.1, target_method = "power_law", target_param = 0.35, step_nb = 10, stat = "mean", precision = "", Rdm_Path = "", QL_Path = ""):
    """
    Main execution function demonstrating the complete analysis pipeline.
    Args:
        target_eps: Epsilon value for the specific configuration to save results
        target_method: Method name for the specific configuration to save results
        target_param: Parameter value for the specific configuration to save results
    """
    # Define Evomol-RL configuration
    ecfp_values = [0, 2]
    eps_values = [0.1, 0.2, 0.3]
    methods_config = {
        "greedy": [0.001, 0.01, 0.1],  # EpsParamList as specified
        "power_law": [0.25, 0.3, 0.35, 0.4],  # EpsParamList as specified
        "constant": [0.1]# Default parameter for constant method
    }
    # Initialize analyzer
    analyzer = AlgorithmAnalyzer(methods_config, ecfp_values, eps_values, 10, "./examples")
    max_realism = float('-inf') #to find max value of %realism
    max_path = []
    
    print("Starting Evomol analysis...")
    # Run Evomol analysis
    evomol_results = analyzer.run_evomol_analysis(analyzer.output_dir+Rdm_Path, precision)
    evomol_stats, max_realism, max_path = analyzer.calculate_statistics(evomol_results, max_realism, max_path, ["Evomol"])
    rdm_steps_reader = ImportData(analyzer.output_dir+Rdm_Path+precision,"sum","EvoMol", 0.1)

    print("\nStarting Evomol-RL analysis...")
    # Run Evomol-RL analysis
    evomol_rl_results = analyzer.run_evomol_rl_analysis(analyzer.output_dir+QL_Path, precision)

    # Calculate statistics for all Evomol-RL configurations
    evomol_rl_stats = {}
    for eps in eps_values:
        evomol_rl_stats[f"eps_{eps}"] = {}
        for method, param_list in methods_config.items():
            evomol_rl_stats[f"eps_{eps}"][method] = {}
            for param in param_list:
                if method == "constant":
                    param = eps
                evomol_rl_stats[f"eps_{eps}"][method][f"{method}_param_{param}"] = {}
                for ecfp in ecfp_values:
                    print(f"Calculating statistics for Evomol-RL: method={method}, param={param}, eps={eps}, ecfp={ecfp}")
                    run_results = evomol_rl_results[f"eps_{eps}"][method][f"{method}_param_{param}"][f"ecfp{ecfp}"]
                    evomol_rl_stats[f"eps_{eps}"][method][f"{method}_param_{param}"][f"ecfp{ecfp}"], max_realism, max_path = analyzer.calculate_statistics(run_results, 
                                                                                                                                                           max_realism, 
                                                                                                                                                           max_path, 
                                                                                                                                                           [eps, method, param, ecfp])
                    # Save intermediate results after each configuration
                    analyzer.save_results_to_file(evomol_rl_stats, f"./examples/EvoMol_RL_Stat{precision}.json")

    print(f"\nMaximum realism percentage found: {max_realism} in configurations: {max_path[0][1]} with param {max_path[0][2]} and eps {max_path[0][0]}")
    
    # Save results and plot for best configuration
    target_eps = max_path[0][0] if max_path else target_eps
    target_method = max_path[0][1] if max_path else target_method
    target_param = max_path[0][2] if max_path else target_param

    if target_eps:    
        target_results = {
            'evomol': {
                'results': evomol_results,
                'statistics': evomol_stats
            },
            'evomol_rl': {
                'config': f"{target_method}{target_param}_eps{target_eps}",
                'results_ecfp0': evomol_rl_results[f"eps_{target_eps}"][target_method][f"{target_method}_param_{target_param}"]["ecfp0"],
                'statistics_ecfp0': evomol_rl_stats[f"eps_{target_eps}"][target_method][f"{target_method}_param_{target_param}"]["ecfp0"],
                'results_ecfp2': evomol_rl_results[f"eps_{target_eps}"][target_method][f"{target_method}_param_{target_param}"]["ecfp2"],
                'statistics_ecfp2': evomol_rl_stats[f"eps_{target_eps}"][target_method][f"{target_method}_param_{target_param}"]["ecfp2"]
            }
        }
        analyzer.save_results_to_file(target_results, f"./examples/{target_method}{target_param}_eps{target_eps}_results.json")

    # Plot realism percent over steps for best configuration
    ql0_steps_reader_eps1 = ImportData(analyzer.output_dir+QL_Path+f"10run_stoch_ecfp0_eps_{target_method}_{target_param}_epsmin_{target_eps}_random_alea_ql_steps500_depth1_C,N,O,F{precision}",
               "sum","ecfp0", target_eps)
    ql2_steps_reader_eps1 = ImportData(analyzer.output_dir+QL_Path+f"10run_stoch_ecfp2_eps_{target_method}_{target_param}_epsmin_{target_eps}_random_alea_ql_steps500_depth1_C,N,O,F{precision}",
                "sum","ecfp2", target_eps)
    
    analyzer.PlotOverSteps(ql0_steps_reader_eps1, ql2_steps_reader_eps1, rdm_steps_reader,
                           'pct_realism',"Evolution of realism percent over steps","Steps" ,"Realism percent", 
                           stat, step_nb, f"{target_method}_{target_param}_eps{target_eps}")                   

    # Create final comparison table
    print("\nCreating comparison table on best configuration...")
    # Create a simplified version with only key metrics for final report
    final_table = analyzer.create_comparison_table(
        evomol_stats, 
        evomol_rl_stats, 
        (target_method, target_param, target_eps),
        metrics=['Pct_realism', 'Pct_novelty']
    )
    # Save final table
    final_table.to_csv(f"./examples/final_comparison_table{precision}.csv")

    # Display summary statistics
    print("\n" + "="*50)
    print("     BEST CONFIGURATION SUMMARY STATISTICS ")
    print("="*50)
    
    print("\nEvomol Results:")
    for metric, stats in evomol_stats.items():
        print(f"{metric}: {stats['mean']:.2f} ± {stats['std']:.2f}")
    
    print(f"\nEvomol-RL Results ({target_method} {target_param}, eps {target_eps}):")
    for ecfp in ecfp_values:
        print(f"\nECFP{ecfp}:")
        target_stats = evomol_rl_stats[f"eps_{target_eps}"][target_method][f"{target_method}_param_{target_param}"][f"ecfp{ecfp}"]#[target_method][target_param][target_eps][ecfp]
        for metric, stats in target_stats.items():
            print(f"{metric}: {stats['mean']:.2f} ± {stats['std']:.2f}")
    
    print(f"\nFinal comparison table format:")
    print("="*80)
    print(final_table)
    print("="*80)

    # Run Kruskal-Wallis test on realism percent between Evomol and Evomol-RL best configuration
    from scipy.stats import kruskal
    Realism_Evomol = [result['Pct_realism'] for result in evomol_results]
    Realism_ecfp0 = [result['Pct_realism'] for result in evomol_rl_results[f"eps_{target_eps}"][target_method][f"{target_method}_param_{target_param}"]["ecfp0"]]
    Realism_ecfp2 = [result['Pct_realism'] for result in evomol_rl_results[f"eps_{target_eps}"][target_method][f"{target_method}_param_{target_param}"]["ecfp2"]]
    stat, p_value = kruskal(Realism_Evomol, Realism_ecfp0, Realism_ecfp2)
    print(f"\nKruskal-Wallis test between Evomol and Evomol-RL (ecfp0 and ecfp2) realism percentages:")
    print("="*90)
    print(f"Statistic: {stat}, p-value: {p_value}")
    if p_value < 0.05:
        if p_value < 0.001: print("The difference in realism percentages is statistically highly significant (p < 0.001).")
        else: print("The difference in realism percentages is statistically significant (p < 0.05).")
        # if Kruskal-Wallis test significant, run post-hoc test
        print(f"\nPost-hoc Wilcoxon tests results:")
        print("="*32)
        from scipy.stats import wilcoxon
        u_stat, p_val_posthoc = wilcoxon(Realism_Evomol, Realism_ecfp0, alternative='two-sided')
        print(f"- between Evomol and Evomol-RL (ecfp0): W-statistic: {u_stat}, p-value: {p_val_posthoc}")
        u_stat, p_val_posthoc = wilcoxon(Realism_Evomol, Realism_ecfp2, alternative='two-sided')
        print(f"- between Evomol and Evomol-RL (ecfp2): W-statistic: {u_stat}, p-value: {p_val_posthoc}")
        u_stat, p_val_posthoc = wilcoxon(Realism_ecfp0, Realism_ecfp2, alternative='two-sided')
        print(f"- between Evomol-RL ecfp 0 and ecfp 2: W-statistic: {u_stat}, p-value: {p_val_posthoc}")
    else:
        print("No statistically significant difference in realism percentages (p >= 0.05).")
    print("="*90)

if __name__ == "__main__":
    main(target_eps = 0.1, 
         target_method = "power_law", 
         target_param = 0.35, 
         step_nb = 10, 
         stat = "sum", 
         precision = "_sillyTh0_best_NopreselestedAct",  
         Rdm_Path = "/Silly_Random/10run_Random_steps500_depth1_C,N,O,F",
         QL_Path = "/Silly_Qlearning/"
   )