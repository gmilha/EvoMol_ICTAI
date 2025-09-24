#########################################################################################################################
#  Silly_random_script.py                                                                                               #
#########################################################################################################################
# Script to run several times the Random optimization with fixed configuration                                          #
#########################################################################################################################
# The optimization aims at minimizing the proportion of sillywalks in the generated                                     #
# molecules.                                                                                                            #
# 10 runs are done for each configuration with different seeds.                                                         #
# The maximum number of steps per run is 500.                                                                           #
# The atoms used are C,N,O,F.                                                                                           #
# The optimization is done with the RandomActionTypeSelectionStrategy and the KRandomGraphOpsImprovingMutationStrategy. #                                                      #
# The initial population is the one in initial_population.smi file containing acetylsalicylic acid                      #
# The selection of molecule to improve is the default one (change the 10 best molecules in the current population)      #
# The valid ECFP are in data/VALID_ECFP/valid_ecfp{ecfp}.json file.                                                     #
# The sillywalks reference database is in data/ECFP/complete_ChEMBL_ecfp4_dict.json file.                               #
# The sillywalks threshold is set to 0 meaning no sillybit admited in generated molecules                               #
#                                                                                                                       #
# The results are saved in ./examples/Silly_Random/ folder.                                                             #
#                                                                                                                       #
# Author : Gaëlle Milon-Harnois                                                                                         #
# Date : May 2025                                                                                                       #                        
#########################################################################################################################

from evomol import run_model, RandomActionTypeSelectionStrategy
from evomol.mutation import KRandomGraphOpsImprovingMutationStrategy
import numpy as np
import random
from joblib import Parallel, delayed
import datetime

n_run = 10
max_depth = 1 
max_steps = 500 
atoms = "C,N,O,F"

# Seed management module
def set_seed(seed):
     random.seed(seed)
     np.random.seed(seed)
    
def main(seed, run_nb):
    model_path = f"./examples/Silly_Random/{n_run}run_Random_steps{str(max_steps)}_depth{str(max_depth)}_{atoms}_sillyTh0_best_NopreselestedAct/run{str(run_nb+1)}"
    set_seed(seed)
    print(datetime.datetime.today())
    run_model({
        "obj_function": "sillywalks_proportion",
        "optimization_parameters": {
            "pop_max_size":5000,
            "problem_type": "min",
            "max_steps": max_steps,
            "mutable_init_pop": False,
            "mutation_max_depth": max_depth,
            "neighbour_generation_strategy": RandomActionTypeSelectionStrategy,
            "improving_mutation_strategy": KRandomGraphOpsImprovingMutationStrategy
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

# Generate 10 test environments with different seed values
# Available seeds list
seeds = [42, 100, 7, 23, 56, 12, 98, 65, 101, 88]

# check run number lower seed list length
if n_run > len(seeds):
    print(f"Erreur : {n_run} run required, but only {len(seeds)} seeds available. Thanks to revise the run number to lower one.")
else:
    print(f"\n--- Running silly_random_script --- ")
    print(" ---- n_run =", n_run , "--- max_depth = ", max_depth, "--- max_steps =" , max_steps, "---", atoms, "--- SillyThreshold = 0")
    ## n_run first seeds selection
    selected_seeds = seeds[:n_run]

    # parallel runs execution
    Parallel(n_jobs=-1)(delayed(main)(seed, run_nb) for run_nb,seed in enumerate(selected_seeds))
    print(f"All {n_run} runs for config depth{max_depth}; sillyTh0 completed.")
    print(datetime.datetime.today())
