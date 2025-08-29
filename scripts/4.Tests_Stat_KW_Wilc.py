import pandas as pd
import numpy as np
from scipy.stats import f_oneway, wilcoxon, kruskal, ttest_ind, shapiro

def ImportData(path, method, eps):
    steps_reader = pd.read_csv(path+"/mean_steps.csv")
    
    #######################  Deletion of line Step 0   #################################""
    steps_reader.drop(steps_reader[steps_reader['n_replaced'] == 0.0 ].index, inplace=True )#index[[0]])
    
    steps_reader["pct_realism"] = 1 - (steps_reader['n_discarded_sillywalks'] / (steps_reader['n_replaced']+steps_reader['n_discarded_sillywalks']+steps_reader['n_discarded_tabu']))
        
    steps_reader.to_csv("./examples/ICTAI/"+method+"_"+str(eps)+".csv", decimal=".", index=True)
    return steps_reader

ql0_steps_reader = ImportData("./examples/ICTAI/10run_stoch_ecfp0_eps_power_law_0.35_epsmin_0.1_random_alea_ql_steps500_depth1_C,N,O,F_sillyTh0",
               "ecfp0", 0.1)
ql2_steps_reader = ImportData("./examples/ICTAI/10run_stoch_ecfp2_eps_power_law_0.35_epsmin_0.1_random_alea_ql_steps500_depth1_C,N,O,F_sillyTh0",
                "ecfp2", 0.1)
rdm_steps_reader = ImportData("./examples/Silly_Random/10run_Random_steps500_depth1_C,N,O,F_RandomPop_250512",
                "EvoMol", 0.1)   

# Données inter steps
group1 = np.array(ql0_steps_reader['pct_realism'])
group2 = np.array(ql2_steps_reader['pct_realism'])
group3 = np.array(rdm_steps_reader['pct_realism'])
print("Group 1 (ecfp0):", group1)
print("Group 2 (ecfp2):", group2)
print("Group 3 (Random):", group3)
# Test ANOVA à un facteur
 #f_oneway(group1, group2,group3)

res = f_oneway(group1, group2, group3)
print(f"One way ANOVA : {res}")

res = kruskal(group1, group2, group3)
print(f"Kruskall-Wallis : {res}")
res = wilcoxon(group1, group2)#, alternative='greater')
print(f"Wilcoxon ECFP0-ECFP2 : {res}")
res = wilcoxon(group1, group3)#, alternative='greater')
print(f"Wilcoxon ECFP0-Random : {res}")
res = wilcoxon(group2, group3)#, alternative='greater')
print(f"Wilcoxon ECFP2-Random : {res}")


# Données inter run

group1 = np.array([0.503968253968254,0.499570425026533,0.510115148954152,0.475056958420353,0.481313212027123,0.511386112150532,
0.492315693667924,0.489604770196823,0.49996166526106,0.508200235823775])
group2 = np.array([0.78167077293302, 0.794477556073825, 0.776247555895605, 0.797070526497895, 0.767115647396748, 0.788615782664942,
0.776142503376857, 0.785850154728444, 0.796799542791827,0.773289108124965])
group3 = np.array([0.827045308764359, 0.824854289312597, 0.818784691790827, 0.831431704353496, 0.831891955929392,
0.836631016042781, 0.837349217276782, 0.812653049465587, 0.840753373869198, 0.837425407475581])
print("Group 1 (ecfp0):", group1)
print("Group 2 (ecfp2):", group2)
print("Group 3 (Random):", group3)
# Test ANOVA à un facteur
 #f_oneway(group1, group2,group3)

Shap1 = shapiro(group1)
print(f"Shapiro-Wilk test for group1 (ecfp0): {Shap1}")
Shap2 = shapiro(group2)
print(f"Shapiro-Wilk test for group2 (ecfp2): {Shap2}")
Shap3 = shapiro(group3)
print(f"Shapiro-Wilk test for group3 (Random): {Shap3}")

res = f_oneway(group1, group2, group3)
print(f"One way ANOVA : {res}")
res_ttest12 = ttest_ind(group1, group2)
print(f"Test de Student ECFP0-ECFP2 : {res_ttest12}")
res_ttest13 = ttest_ind(group1, group3)
print(f"Test de Student ECFP0-Random : {res_ttest13}")
res_ttest23 = ttest_ind(group2, group3)
print(f"Test de Student ECFP2-Random : {res_ttest23}")

resKruskal = kruskal(group1, group2, group3)
print(f"Kruskall-Wallis : {resKruskal}")
resWilc12 = wilcoxon(group1, group2)#, alternative='greater')
print(f"Wilcoxon ECFP0-ECFP2 : {resWilc12}")
resWilc13 = wilcoxon(group1, group3)#, alternative='greater')
print(f"Wilcoxon ECFP0-Random : {resWilc13}")
resWilc23 = wilcoxon(group2, group3)#, alternative='greater')
print(f"Wilcoxon ECFP2-Random : {resWilc23}")
