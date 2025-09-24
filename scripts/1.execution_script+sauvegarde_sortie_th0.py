##########################################################################################
# Script to execute EvoMol and Q-learning scripts and save their outputs in a text file  #
##########################################################################################
# Author : Gaëlle Milon-Harnois                                                          #
# Date : April 2025                                                                      #
##########################################################################################
import subprocess
import sys
import datetime

date = str(datetime.date.today())
path = "scripts/"
file = "Random + qlearning_script_n_parallel_run_several_config_th0"
file_EvoMol ="Silly_random_script"
#file_EvoMolRL = "qlearning_script_n_parallel_run_several_config_th0" 

# script execution. Log saved in a txt file
with open(f'output_{file}_{date}.txt', 'w') as f:
   subprocess.run([sys.executable, f'{path}{file_EvoMol}.py'], stdout=f, stderr=f)
   #subprocess.run([sys.executable, f'{path}{file_EvoMolRL}.py'], stdout=f, stderr=f)