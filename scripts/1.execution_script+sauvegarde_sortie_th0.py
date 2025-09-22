import subprocess
import sys
import datetime

date = str(datetime.date.today())
path = "scripts/"
file = "qlearning_script_n_parallel_run_several_config_th0" 

# script execution. Log saved in a txt file
with open(f'output_{file}_{date}.txt', 'w') as f:
   subprocess.run([sys.executable, f'{path}{file}.py'], stdout=f, stderr=f)