import subprocess
import sys
import datetime

date = str(datetime.date.today())
path = "scripts/"
file = "qlearning_script_n_parallel_run_several_config_th0" #"qlearning_script_n_parallel_run"#"qlearning_script_n_run" #"qlearning_random_select_script_n_run" #'Silly_random_script' ##"qed_script" 
#"qed_qlearning_script" #'scripts/qlearning_script_n_run.py' / 'Plot_10_run_stochastic_results_Sparse_w.py' / 
# 'scripts/random_script.py'

# Exécuter le script et capturer la sortie dans un fichier
with open(f'output_{file}_{date}.txt', 'w') as f:
   subprocess.run([sys.executable, f'{path}{file}.py'], stdout=f, stderr=f)