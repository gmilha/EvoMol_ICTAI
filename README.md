# EvoMol and Emomol-RL


[EvoMol](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00458-z#citeas) is a flexible and interpretable
evolutionary algorithm designed for molecular properties optimization. It can optimize any (customizable) objective
function. It can also maximize the
[diversity of generated molecules](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00554-8).

EvoMol-RL is a significant extension of the EvoMol evolutionary algorithm that integrates reinforcement learning to guide molecular mutations based on local structural context.
---

## Installation

EvoMol was designed on Ubuntu (18.04+). Some features might be missing on other systems. Especially, the drawing of
exploration trees is currently unavailable on Windows.

To install EvoMol on your system, run the following commands in your terminal. The installation depends on
<a href='https://www.anaconda.com/products/individual'>Anaconda</a>.

```shell script
$ git clone https://github.com/gmilha/EvoMol_ICTAI.git    # Clone EvoMol
$ cd EvoMol                                               # Move into EvoMol directory
$ conda env create -f evomol_env.yml                      # Create conda environment
$ conda activate evomolenv                                # Activate environment
$ python -m pip install .                                 # Install EvoMol
```

---

## Quickstart

Launching a <a href="https://www.nature.com/articles/nchem.1243">QED</a> optimization for 500 steps. Beware, you need to
activate the evomolenv conda environment when you use EvoMol.

```python
from evomol import run_model
run_model({
    "obj_function": "qed",
    "optimization_parameters": {
        "max_steps": 500
    },
    "io_parameters": {
        "model_path": "examples/1_qed"
    },
})
```

---

## Settings

A dictionary can be given to evomol.run_model to describe the experiment to be performed. This dictionary can contain up 
to 4 entries, that are described in this section.

**Default values** are represented in bold.

### Objective function

The ```"obj_function"``` attribute can take the following values. Multi-objective functions can be nested to any depth. 
* Implemented functions: 
  * "<a href="https://www.nature.com/articles/nchem.1243">qed</a>",
    "<a href="https://arxiv.org/abs/1610.02415v2">plogp</a>",
    "<a href="https://www.nature.com/articles/s41598-019-47148-x">norm_plogp</a>",
    "<a href="https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-1-8">sascore</a>",
    "<a href="https://arxiv.org/abs/1705.10843">norm_sascore</a>",
    "<a href="https://www.frontiersin.org/articles/10.3389/fchem.2020.00046/full">clscore</a>".
  * "<a href=https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839>isomer</a>_formula" (*e.g.* "isomer_C7H16").
  * "<a href=https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839>rediscovery</a>_smiles" (*e.g.* "rediscovery_CC(=O)
    OC1=CC=CC=C1C(=O)O")
  * "homo", "lumo", "gap", "homo-1"
  * "entropy_ifg", "entropy_gen_scaffolds", "entropy_shg_1", "entropy_checkmol" and "entropy_ecfp4" can be used to
    maximize the entropy of descriptors, respectively
    using <a href="https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0225-z">IFGs </a>,
    <a href="https://pubs.acs.org/doi/10.1021/jm9602928">Murcko generic scaffolds</a>, level 1
    <a href="https://link.springer.com/article/10.1186/s13321-018-0321-8">shingles</a>,
    <a href="https://homepage.univie.ac.at/norbert.haider/cheminf/cmmm.html">checkmol</a> and ECFP4 fingerprints (RDKit
    Morgan fingerprints implementation).
  * "n_perturbations": count of the number of perturbations that were previously applied on the molecular graph during
    the optimization. If the "mutation_max_depth" parameter is set to 1, then this is equivalent to the number of
    mutations.
  * "sillywalks_proportion": proportion of ECFP4 features that failed the
    [sillywalks filter](https://github.com/PatWalters/silly_walks), based on the reference dataset given in the
    ```"silly_molecules_reference_db_path"``` parameter.
* A custom function evaluating a SMILES. It is also possible to give a tuple (function, string function name).
* A dictionary describing a multi-objective function and containing the following entries (see the [example section](https://github.com/jules-leguy/EvoMol#Designing-complex-objective-functions)).
    * ```"type"``` : 
      * "linear_combination" (linear combination of the properties).
      * "product" (product of the properties).
      * "mean" (mean of the properties).
      * "abs_difference" (absolute difference of **exactly 2** properties).
    * ```"functions"``` : list of functions (string keys describing implemented functions, custom functions,
    multi-objective functions or wrapper functions).
    * Specific to the linear combination
        * ```"coef"``` : list of coefficients.
* A dictionary describing a function wrapping a single property and containing the following entries (see the [example section](https://github.com/jules-leguy/EvoMol#Designing-complex-objective-functions)).
  * ```"type"```:
     * "gaussian" (passing the value of a unique objective function through a Gaussian function).
     * "opposite" (computing the opposite value of a unique objective function).
     * "sigm_lin", (passing the value of a unique objective through a linear function and a sigmoid function).
     * "one_minus" (computing 1-f(x) of a unique objective function f).
  * ```"function"``` the function to be wrapped (string key describing an implemented function, custom function,
  multi_objective function or wrapper function). For compatibility reasons, it is also possible to use a 
  ```"functions"``` attribute that contains a list of functions. In that case only the first element of the list is
  considered.
  * Specific to the use of a Gaussian function
    * ```"mu"```: μ parameter of the Gaussian.
    * ```"sigma"```: σ parameter of the Gaussian.
    * ```"normalize"```: whether to normalize the function so that the maximum value is exactly 1 (**False**).
  * Specific to the use of sigmoid/linear functions
      * ```"a"``` list of *a* coefficients for the *ax+b* linear function definition.
      * ```"b"``` list of *b* coefficients for the *ax+b* linear function definition.
      * ```"lambda"``` list of *λ* coefficients for the sigmoid function definition.
* An instance of evomol.evaluation.EvaluationStrategyComposant
* ```"guacamol_v2"``` for taking the goal directed <a href="https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839">
GuacaMol</a> benchmarks.


### Search space

The ```"action_space_parameters"``` attribute can be set with a dictionary containing the following entries.
* ```"atoms"``` : text list of available <ins>heavy</ins> atoms (**"C,N,O,F,P,S,Cl,Br"**).
* ```"max_heavy_atoms"```: maximum molecular size in terms of number of heavy atoms (**38**).
* ```"append_atom"```: whether to use *append atom* action (**True**).
* ```"remove_atom"```: whether to use *remove atom* action (**True**).
* ```"change_bond"```: whether to use *change bond* action (**True**).
* ```"change_bond_prevent_breaking_creating_bonds"```: whether to prevent the removal or creation of bonds by *change_bond* action (**False**) 
* ```"substitution"```: whether to use *substitute atom type* action (**True**).
* ```"cut_insert"```: whether to use *cut atom* and *insert carbon atom* actions (**True**).
* ```"move_group"```: whether to use *move group* action (**True**).
* ```"remove_group"```: whether to use *remove group* action (**False**).
* ```"remove_group_only_remove_smallest_group"```: in case remove group action is enabled, whether to be able to remove 
* both parts of a bridge bond (False), or only the smallest part in number of atoms (**True**).
* ```"use_rd_filters"```: whether to use the <a href=https://github.com/PatWalters/rd_filters>rd_filter program</a> as a 
quality filter before inserting the mutated individuals in the population (**False**).
* ```"sillywalks_threshold``` maximum proportion of [silly bits](https://github.com/PatWalters/silly_walks) in the ECFP4 
fingerprint of the solutions with respect to a reference dataset (see IO parameters). If the proportion is above the
threshold, the solutions will be discarded and thus will not be inserted in the population (**1**).
* ```"sascore_threshold"``` if the solutions have
  a [SAScore](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-1-8)
  value above this threshold, they will be discarded and thus will not be inserted in the population (**float("inf")**).
* ```"custom_filter_function"```: custom boolean function evaluating a SMILES that defines a filter on the accessible
  search space. If the result is ```True```, the molecule is considered valid, and otherwise it is considered invalid
  (**None**).
* ```"sulfur_valence"```: valence of sulfur atoms (**6**).

### Optimization parameters

The ```"optimization_parameters"``` attribute can be set with a dictionary containing the following entries.
* ```"pop_max_size"``` : maximum population size (**1000**).
* ```"max_steps"``` : number of steps to be run before stopping EvoMol (**1500**).
* ```"max_obj_calls""```: number of calls to the objective functions before stopping EvoMol (**float("inf")**).
* ```"stop_kth_score_value"```: stopping the search if the kth score in descendant value order has reached the given 
value with given precision. Accepts a tuple (k, score, precision) or **None** to disable.
* ```"k_to_replace"``` : number of individuals replaced at each step (**10**).
* ```"selection"``` : whether the best individuals are selected to be mutated (**"best"**), or they are selected 
randomly with uniform distribution ("random"), or they are selected randomly with a probability that is proportional
to their objective function value ("random_weighted") .
* ```"problem_type"``` : whether it is a maximization (**"max"**) or minimization ("min") problem.
* ```"mutation_max_depth"``` : maximum number of successive actions on the molecular graph during a single mutation 
(**2**).
* ```"neighbour_generation_strategy"```: strategy to generate neighbour candidates (mutation). By default, the type of
perturbation is first drawn randomly, then the actual perturbation of previously selected type is drawn randomly among 
the valid ones (**evomol.molgraphops.exploration.RandomActionTypeSelectionStrategy()**).
* ```"mutation_find_improver_tries"``` : maximum number of mutations to find an improver (**50**).
* ```"guacamol_init_top_100"``` : whether to initialize the population with the 100 best scoring individuals of the 
GuacaMol <a href="https://academic.oup.com/nar/article/45/D1/D945/2605707">ChEMBL</a> subset in case of taking the 
GuacaMol benchmarks (**False**). The list of SMILES must be given as initial population.
* ```"mutable_init_pop"``` : if True, the individuals of the initial population can be freely mutated. If False, they 
can be branched but their atoms and bonds cannot be modified (**True**).
* ```"n_max_desc"```: max number of descriptors to be possibly handled when using an evaluator relying on a vector of 
descriptors such as entropy contribution (**3.000.000**).
* ```"shuffle_init_pop"```: whether to shuffle the smiles at initialization (**False**).

### Input-Output parameters

The ```"io_parameters"``` attribute can be set with a dictionary containing the following entries.
* ```"model_path"``` : path where to save model's output data (**"EvoMol_model"**).
* ```"smiles_list_init"```: list of SMILES describing the initial population (**None**: interpreting the
  ```"smiles_list_init_path"``` attribute). Note : not available when taking GuacaMol benchmarks.
* ```"smiles_list_init_path"``` : path where to find the SMILES list text file describing the initial population (one
  SMILES per row). It is also possible to pass the path to the *pop.csv* file from a previous EvoMol experiment. In the
  latter case, the population will be initialized as it was at the end of the loaded experiment
  (**None**: initialization of the population with a single methane molecule).
* ```"external_tabu_list"```: list of SMILES that won't be generated by EvoMol.
* ```"record_history"``` : whether to save exploration tree data. Must be set to True to later draw the exploration tree
 (**False**).
* ```"record_all_generated_individuals"``` : whether to record a list of all individuals that are generated during the 
entire execution (even if they fail the objective function computation or if they are not inserted in the population as
they are not improvers). Also recording the step number and the total number of calls to the objective function at the 
time of generation (**False**).
* ```"save_n_steps"``` : period (steps) of saving the data (**100**).
* ```"print_n_steps"``` : period (steps) of printing current population statistics (**1**).
* ```"dft_working_dir"``` : path where to save DFT optimization related files (**"/tmp"**).
* ```"dft_cache_files"``` : list of json files containing a cache of previously computed HOMO or LUMO values (**[]**).
* ```"dft_MM_program"``` : program used to compute molecular mechanics initial geometry of DFT calculations. The 
options are :
  * "**obabel_mmff94**" or "obabel" to combine OpenBabel and the MMFF94 force field.
  * "rdkit_mmff94" to combine RDKit with the MMFF94 force field.
  * "rdkit_uff" to combine RDKit with the UFF force field.
* ```"dft_base"```: DFT calculations base (__"3-21G*"__).
* ```"dft_method"``` : DFT calculations method (**B3LYP**).
* ```"dft_n_jobs"```: number of threads assigned to each DFT calculation (**1**).
* ```"dft_mem_mb"```: memory assigned to each DFT calculation in MB (**512**).
* ```"silly_molecules_reference_db_path``` : path to a JSON file that represents a dictionary containing as keys all the
  ECFP4 bits that are extracted from a reference dataset of quality solutions (**None**). See the
  ```"sillywalks_threshold"``` parameter.
* ```"evaluation_strategy_parameters"``` : a dictionary that contains an entry "evaluate_init_pop" to set given 
parameters to the EvaluationStrategy instance in the context of the evaluation of the initial population. An entry
 "evaluate_new_sol" must be also contained to set given parameters for the evaluation of new solutions during the 
 optimization process. If None, both keys are set to an empty set of parameters (**None**).


---

## Citing EvoMol

To reference EvoMol, please cite one of the following articles.

Leguy, J., Cauchy, T., Glavatskikh, M., Duval, B., Da Mota, B. EvoMol: a flexible and interpretable evolutionary
algorithm for unbiased de novo molecular generation. J Cheminform 12, 55 (2020).
https://doi.org/10.1186/s13321-020-00458-z

Leguy, J., Glavatskikh, M., Cauchy, T. et al. Scalable estimator of the diversity for de novo molecular generation
resulting in a more robust QM dataset (OD9) and a more efficient molecular optimization. J Cheminform 13, 76 (2021). https://doi.org/10.1186/s13321-021-00554-8
