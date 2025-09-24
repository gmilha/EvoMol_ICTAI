#####################################################################################################################################
# qlearning_script_n_parallel_run_several_config_th0.py                                                                             #
#####################################################################################################################################
# Script to run several times the Q-learning optimization with different configurations                                             #
# (ecfp, epsilon_method, EpsParam, epsilon_min) and paralelization of n_run runs                                                    #
#####################################################################################################################################
# Configurations tested :                                                                                                           #
# - epsilon_min : 0.1, 0.2, 0.3                                                                                                     #
# - max_depth : 1                                                                                                                   #
# - ecfp : 0, 2                                                                                                                     #
# - epsilon_method and EpsParam :                                                                                                   #
#     - epsilon_method = "greedy"; EpsParam : 0.001, 0.01, 0.1                                                                      #
#     - epsilon_method = "power_law"; EpsParam : 0.25, 0.3, 0.35, 0.4                                                               #
#     - epsilon_method = "constant"; EpsParam : epsilon_min                                                                         #
#####################################################################################################################################
# The optimization aims at minimizing the proportion of sillywalks in the generated                                                 #
# molecules.                                                                                                                        #
# 10 runs are done for each configuration with different seeds.                                                                     #
# The maximum number of steps per run is 500.                                                                                       #
# The atoms used are C,N,O,F.                                                                                                       #
# The optimization is done with the StochasticQLearningActionSelectionStrategy and the QLearningGraphOpsImprovingMutationStrategy.  #                                                       #
# The initial population is the one in initial_population.smi file containing acetylsalicylic acid                                  #
# The selection of molecule to improve is the default one (change the 10 best molecules in the current population)                  #
# The valid ECFP are in data/VALID_ECFP/valid_ecfp{ecfp}.json file.                                                                 #
# The sillywalks reference database is in data/ECFP/complete_ChEMBL_ecfp4_dict.json file.                                           #
# The sillywalks threshold is set to 0 meaning no sillybit admited in generated molecules                                           #
#                                                                                                                                   #
# The results are saved in ./examples/Silly_Qlearning/ folder.                                                                      #
#                                                                                                                                   #
# Author : Gaëlle Milon-Harnois                                                                                                     #
# Date : May 2025                                                                                                                   #    
#####################################################################################################################################

from evomol import run_model
from evomol.mutation import QLearningGraphOpsImprovingMutationStrategy
from evomol.molgraphops.exploration import StochasticQLearningActionSelectionStrategy
import numpy as np
import random 
from joblib import Parallel, delayed
import datetime

n_run = 10 
max_depth = 1 
max_steps = 500 
atoms = "C,N,O,F"
epsilon_0 = 1.0
p0 = 0.2
gamma = 0.95

def set_seed(seed):
     random.seed(seed)
     np.random.seed(seed)  

def main(seed, run_nb, ecfp, epsilon_method, EpsParam, epsilon, max_depth):
    model_path = f"./examples/Silly_Qlearning/{n_run}run_stoch_ecfp{ecfp}_eps_{epsilon_method}_{EpsParam}_epsmin_{epsilon}_random_alea_ql_steps{str(max_steps)}_depth{str(max_depth)}_{atoms}_sillyTh0_best_NopreselestedAct/run{str(run_nb+1)}"
    set_seed(seed)
    print(datetime.datetime.today())
    run_model({
        "obj_function": "sillywalks_proportion",
        "ql_parameters": {
            "ecfp": ecfp,
            "alpha": EpsParam,
            "epsilon": epsilon,
            "epsilon_min": epsilon,
            "epsilon_0": epsilon_0,
            "lambd": EpsParam,
            "epsilon_method": epsilon_method,
            "disable_updates": False,
            "record_trained_weights": True,
            "init_weights_file_path": None,
            "valid_ecfp_file_path": "data/VALID_ECFP/valid_ecfp" + str(ecfp) + ".json",
        },
        "optimization_parameters": {
            "pop_max_size":5000,
            "problem_type": "min",
            "max_steps": max_steps,
            "mutable_init_pop": False,
            "mutation_max_depth": max_depth,
            "neighbour_generation_strategy": StochasticQLearningActionSelectionStrategy,
            "improving_mutation_strategy": QLearningGraphOpsImprovingMutationStrategy
        },
        "io_parameters": {
            "model_path": model_path,
            "save_n_steps": 1,
            "record_history": True,
            "record_all_generated_individuals": True,
            "smiles_list_init_path": "initial_population.smi", 
            "silly_molecules_reference_db_path": "data/ECFP/complete_ChEMBL_ecfp4_dict.json"
        },
        "action_space_parameters": {
            "atoms": atoms,
            "substitution": False,
            "cut_insert": False,
            "move_group": False,
            "remove_group": False,
            "sillywalks_threshold": 0
        },
     })

# Manage 10 testing environnements with différent seeds
# List of available seeds
seeds = [42, 100, 7, 23, 56, 12, 98, 65, 101, 88]

# check count of run is lower than Seeds list length
if n_run > len(seeds):
    print(f"Error : {n_run} runs requested, but only {len(seeds)} seeds available. Please lower the number of runs")
else:
    print(f"\n--- Running qlearning_script --- ")
    ## Select n_run first seeds listed
    selected_seeds = seeds[:n_run]

for epsilon in {0.1, 0.2, 0.3}:
    for ecfp in {2, 0} :
        for epsilon_method in ["greedy", "power_law", "constant"]:
            if epsilon_method == "greedy":
                EpsParamList = [0.001, 0.01, 0.1]
            elif epsilon_method == "power_law":
                EpsParamList = [0.25, 0.3, 0.35, 0.4] 
            else: #epsilon_method == "constant":
                EpsParamList = [epsilon]

            for EpsParam in EpsParamList:

                print("ecfp", ecfp, " --- Epsilon =", epsilon," epsilon",epsilon_method,EpsParam,"--- n_run =", n_run , "--- max_depth = ", max_depth, "--- max_steps =" , max_steps, "---", atoms, "--- SillyThreshold = 0")
                # paralelization of n_run with joblib
                Parallel(n_jobs=-1)(delayed(main)(seed, run_nb, ecfp, epsilon_method, EpsParam, epsilon, max_depth) for run_nb,seed in enumerate(selected_seeds))
                print(f"All {n_run} runs for config depth{max_depth}; sillyTh0; ecfp{ecfp}; epsilon{epsilon_method}_{EpsParam}; epsilonmin{epsilon} completed.")
                print(datetime.datetime.today())