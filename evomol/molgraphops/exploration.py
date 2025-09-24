import json
from abc import ABC, abstractmethod

import numpy as np
import random

from rdkit.Chem import Kekulize, MolFromSmiles
from rdkit import Chem  
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

from os import makedirs
from os.path import dirname, join

from evomol.molgraphops.molgraph import MolGraphBuilder, MolGraph
from evomol.molgraphops.actionspace import ActionSpace
from evomol.notification import Observer
from scipy.sparse import csr_matrix, coo_matrix, vstack

from collections import defaultdict

def _compute_root_node_id():
    return ""


def _compute_new_edge_name(action_coords):
    """
    Computing the name of the edge created by applying the given action to the given state of the molecular graph
    The edge name is the concatenation of the action type and the action id
    @param action_coords:
    @return:
    """
    return str(action_coords[0]) + "-" + str(action_coords[1])


def _compute_new_node_id(parent_node_id, action_coords):
    """
    Computing the identifier of a node from the action coordinates and the identifier of its parent.
    The node id is the concatenation of the id of its parent and the name of its edge with its parent (action)
    :param parent_node_id:
    :param action_coords
    :return:
    """

    if parent_node_id == _compute_root_node_id():
        separator = ""
    else:
        separator = "_"

    return parent_node_id + separator + _compute_new_edge_name(action_coords)


def random_neighbour(molgraph_builder, depth, return_mol_graph=False, uniform_action_type=True):
    """
    Computing a random neighbour of depth level
    Returning a tuple (SMILES, id) or (mol. graph, id)
    Raising an exception if no neighbour of given depth was found
    @param depth:
    @param molgraph_builder:
    @param return_mol_graph: whether to return a QuMolGraph object or a SMILES
    @param uniform_action_type: If true, the action type is drawn with a uniform law before the action is drawn. If
    false, the action is drawn directly with uniform law among all possible actions
    """

    # Initialization of molecular graph ID
    id = _compute_root_node_id()

    # Copying and sanitizing (kekulize) QuMolGraphBuilder
    molgraph_builder = molgraph_builder.copy()

    for i in range(depth):
        # Valid action list initialization
        valid_action_coords_list = []

        if uniform_action_type:
            # Drawing the action space
            action_space_k = np.random.choice(molgraph_builder.get_action_spaces_keys())

            # Computing the mask of valid actions
            action_space_mask = molgraph_builder.get_valid_mask_from_key(action_space_k)

            # Extracting the set of valid actions
            valid_actions = np.nonzero(action_space_mask)

            # Creating the set of valid action coords
            for valid_act in valid_actions[0]:
                valid_action_coords_list.append((action_space_k, int(valid_act)))

        else:
            # Computing valid actions
            valid_action_dict = molgraph_builder.get_action_spaces_masks()

            # Iterating over the actions of the different action spaces
            for key, validity in valid_action_dict.items():
                # Recording the id of the valid actions for the current action space
                curr_key_valid_actions = np.nonzero(validity)

                # Iterating over the valid actions for the current action space
                for curr_key_valid_act in curr_key_valid_actions[0]:
                    # Adding the current valid action to the list
                    valid_action_coords_list.append((key, int(curr_key_valid_act)))

        if valid_action_coords_list:
            # Drawing action to apply
            rand_valid_act = np.random.choice(np.arange(len(valid_action_coords_list)))
            action_coords = valid_action_coords_list[rand_valid_act]
            # Updating molecule ID
            id = _compute_new_node_id(id, action_coords)
            # Applying action
            molgraph_builder.execute_action_coords(action_coords)

    if return_mol_graph:
        return molgraph_builder.qu_mol_graph, id
    else:
        return molgraph_builder.qu_mol_graph.to_aromatic_smiles(), id

class NeighbourGenerationStrategy(ABC):
    """
    Strategy that defines how neighbour solutions are generated.
    Either a neighbour is selected randomly with uniform low from the set of all possible valid neighbours
    (preselect_action_type=False), either the type of perturbation/mutation is selected first and then the action is
    selected randomly with uniform law among valid neighbours from selected perturbation type
    (preselect_action_type=True). In the latter case, the implementations of this class define how the action type is
    selected.
    """

    def __init__(self, preselect_action_type=False):
        """
        :param preselect_action_type: whether to first select the action type and then select the actual valid
        perturbation of selected type (True), or whether to select the actual perturbation among all possible ones of
        all types (False).
        """
        self.preselect_action_type = preselect_action_type

    @abstractmethod
    def select_action_type(self, action_types_list, evaluation_strategy):
        """
        Selection of the action type.
        :param action_types_list: list of available action types
        :param evaluation_strategy: instance of evomol.evaluation.EvaluationStrategyComposite
        :return: a single selected action type
        """
        pass

    def generate_neighbour(self, molgraph_builder, depth, evaluation_strategy, return_mol_graph=False):
        """
        :param molgraph_builder: evomol.molgraphops.molgraph.MolGraphBuilder instance previously set up to apply
        perturbations on the desired molecular graph.
        :param depth in number of perturbations of the output neighbour.
        :param evaluation_strategy: evomol.evaluation.EvaluationStrategyComposite instance that is used to evaluate the
        solutions in the EvoMol optimization procedure
        :param return_mol_graph: whether to return the molecular graph (evomol.molgraphops.molgraph.MolGraph) or a
        SMILES.
        :return: (evomol.molgraphops.molgraph.MolGraph, string id of the perturbation) or
        (SMILES, string id of the perturbation)
        """
        # Initialization of molecular graph ID
        id = _compute_root_node_id()

        # Copying and sanitizing (kekulize) QuMolGraphBuilder
        molgraph_builder = molgraph_builder.copy()

        for i in range(depth):
            # Valid action list initialization
            valid_action_coords_list = []

            # The perturbation type is selected before selecting the actual perturbation
            if self.preselect_action_type:
                # Selecting the action type
                action_space_k = self.select_action_type(molgraph_builder.get_action_spaces_keys(), evaluation_strategy)

                # Computing the mask of valid actions
                action_space_mask = molgraph_builder.get_valid_mask_from_key(action_space_k)

                # Extracting the set of valid actions
                valid_actions = np.nonzero(action_space_mask)

                # Creating the set of valid action coords
                for valid_act in valid_actions[0]:
                    valid_action_coords_list.append((action_space_k, int(valid_act)))

            # The perturbation is selected randomly from the set of all possible perturbations
            else:
                # Computing valid actions
                valid_action_dict = molgraph_builder.get_action_spaces_masks()

                # Iterating over the actions of the different action spaces
                for key, validity in valid_action_dict.items():
                    # Recording the id of the valid actions for the current action space
                    curr_key_valid_actions = np.nonzero(validity)
                    # Iterating over the valid actions for the current action space
                    for curr_key_valid_act in curr_key_valid_actions[0]:
                        # Adding the current valid action to the list
                        valid_action_coords_list.append((key, int(curr_key_valid_act)))

            if valid_action_coords_list:
                # Drawing action to apply
                rand_valid_act = np.random.choice(np.arange(len(valid_action_coords_list)))
                action_coords = valid_action_coords_list[rand_valid_act]
                # Updating molecule ID
                id = _compute_new_node_id(id, action_coords)
                # Applying action
                molgraph_builder.execute_action_coords(action_coords)

        if return_mol_graph:
            return molgraph_builder.qu_mol_graph, id
        else:
            return molgraph_builder.qu_mol_graph.to_aromatic_smiles(), id


class RandomActionTypeSelectionStrategy(NeighbourGenerationStrategy):
    """
    Selection of the action type randomly with uniform law.
    """
    def select_action_type(self, action_types_list, evaluation_strategy):
        rand_act_chosen = np.random.choice(action_types_list) 
        return rand_act_chosen

class AlwaysFirstActionSelectionStrategy(NeighbourGenerationStrategy):
    """
    Always selecting the first action type
    """

    def select_action_type(self, action_types_list, evaluation_strategy):
        return action_types_list[0]

class QLearningActionSelectionStrategy(NeighbourGenerationStrategy, Observer, ABC):
    """
    Selection of the action type according to a Q-learning strategy.
    """
    def __init__(self, depth, number_of_accepted_atoms, ecfp, valid_ecfp_file_path=None,
                 init_weights_file_path=None, preselect_action_type=False, disable_updates=False):
        """
        :param depth: number of consecutive executed actions before evaluation
        :param number_of_accepted_atoms: number of accepted atoms in the molecule
        :param ecfp: ECFP number considered involving the diameter considered arround the atom
        :param valid_ecfp_file_path: path to the file containing the valid ECFPs
        :param init_weights_file_path: initial weights for the Q-learning strategy
        :param preselect_action_type: whether to preselect the action type
        :param disable_updates: whether to disable the updates of the Q-learning strategy
        before selecting the actual action
        """
        super().__init__(preselect_action_type)
        # Initializing the depth of the search
        self.depth = depth
        self.depth_counter = 0
        # Initializing which ECFP to search for
        self.ecfp = ecfp
        # Initializing the array of valid ECFP
        self.valid_ecfps = self.Load_Valid_ecfps(valid_ecfp_file_path)
        # Number of valid contexts base on the number of the valid molecules ECFP-0
        self.number_of_contexts = self.valid_ecfps.shape[0]
       
        # Initializing the weights for each action type
        self.Weights = self.initialize_weights(init_weights_file_path, number_of_accepted_atoms, ['AddA', 'RmA', 'ChB'])
        self.step = 0

        # Feature vector of the current state needed for the update of the weights
        self.current_features = []

        # Whether to disable the update of the weights
        self.disable_updates = disable_updates

    def Load_Valid_ecfps(self, file_path):
        """
        Extracting the valid ECFP from the specified json file
        :param file_path: path to the file containing the valid ECFPs
        :return: list of valid ECFP
        """
        if file_path is None:
            raise ValueError('The path to the file containing the valid ECFPs is not specified.')
        with open(file_path, 'r') as f:
            return np.array(json.load(f))
            
    def extract_features(self, molgraph_builder, ecfps, valid_action, action_space):
        """
        Getting the ECFP where the action will be applied and translating it into a float vector of valid ECFP
        :param molgraph_builder: molecule graph builder
        :param ecfps: list of ECFP of the molecule
        :param valid_action: valid action to apply
        :param action_space: action space
        :return: feature vector of the current state for a given action
        """
        qry_ecfp = set()
        CurrFeatDecod = {}
        mol = molgraph_builder.qu_mol_graph
        context = action_space.get_context(valid_action[1], molgraph_builder.parameters, qu_mol_graph=mol)
         
        qry_ecfp.add(ecfps[context[0]][0]) #ecfp position considered atom (left atom if chB)
        ContextNb = self.number_of_contexts+1
                
        if valid_action[0] == 'AddA':
            #index contains the position of ecfp*action
            #position + init_index (depending added atom)
            index = (context[1]*ContextNb) + ecfps[context[0]][1]
            #CurrFeatDecod contains for AddA for each position in contexts_weights: ecfp, ecfp_position, atom position in mol, 
            # type of atom in mol, idx_atom_to_add, type of atom to add 
            CurrFeatDecod = {index:{'ecfp_atom':ecfps[context[0]][0], 
                                'ecfp_init_idx':ecfps[context[0]][1], 
                                'Chosen_action':valid_action,
                                'Mol_SMILES': molgraph_builder.qu_mol_graph.to_aromatic_smiles(), 
                                'atom position':context[0], 
                                'atom type':mol.get_atom_type(context[0]), 
                                'Added atom #':context[1], 
                                'Added atom type':molgraph_builder.parameters.accepted_atoms[context[1]]}}#
            
        elif valid_action[0] == 'RmA':
            index = (ecfps[context[0]][1]) #ecfp position
            #CurrFeatDecod contains for RmA for each position in contexts_weights: ecfp, atom position in mol, type of atom in mol
            CurrFeatDecod = {index:{'ecfp_atom':ecfps[context[0]][0],
                                'Chosen_action':valid_action,
                                'Mol_SMILES': molgraph_builder.qu_mol_graph.to_aromatic_smiles(),
                                'atom position':context[0],
                                'atom type':mol.get_atom_type(context[0])}}
            
        elif valid_action[0] == 'ChB':
            qry_ecfp.add(ecfps[context[1]][0]) #ecfp position Right atom
            index_left = (ecfps[context[0]][1]) #return the index corresponding to the ecfp of the atom in context[0] for the index corresponding to context[0]+1 (+1 due to 1st position = ecfp #)
            index_right = (ecfps[context[1]][1])
            #keep both positions in ascending order
            index_ALL = [min(index_left, index_right), max(index_left, index_right)]
            init_index = ContextNb*ContextNb*(context[3]) 
            ContextMinEcfp = (index_ALL[0]*ContextNb)
            index = init_index + ContextMinEcfp + index_ALL[1] 
            
            #CurrFeatDecod contains for ChB for each position in contexts_weights: ecfp_left_atom,  ecfp_right_atom, /
            # left atom position in mol, right atom position in mol, type of left atom in mol, type of right atom in mol, /
            # current bond, new bond            
            CurrFeatDecod = {index:{'ecfp_atom(left, right)':(ecfps[context[0]][0], ecfps[context[1]][0]), 
                                'Chosen_action':valid_action,
                                'Mol_SMILES': molgraph_builder.qu_mol_graph.to_aromatic_smiles(),
                                'atom position':(context[0], context[1]), 
                                'atom_type(left, right)': (mol.get_atom_type(context[0]), mol.get_atom_type(context[1])),
                                'Current_Bond':context[2], 
                                'New_Bond':context[3]}}
            
        else:
           raise ValueError('Invalid action')
            
        return index, CurrFeatDecod
        
    def Submol_Smiles(self, mol,rad,atom_idx, atom_ids):
        """
        Editing the SMILES of a submol corresponding to the ecfp diameter
        :param mol: molecule to consider
        :param rad: radius considered int(ecfp/2)
        :param atom_idx: position of the considered atom in the mol
        :param atom_ids: positions of the considered ecfp in the mol
        :return: SMILES of the considered atom
        """
        if rad != 0:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol,rad,atom_idx) #renvoi les arrêtes (bond)
            # pour récupérer bonds + atomes utiliser GetBondWithIdx
            print ("Environment atom pos",atom_idx," :",env)
            amap={}
            submol=Chem.PathToSubmol(mol,env,atomMap=amap) 
            #amap corresponds to a dict centered on the considered atom containing 
            # the index of the environmental atom as key and the position on the environement as value           
        else: 
            submol = mol.GetAtomWithIdx(atom_ids[0])
            
        # Obtenir le SMILES du sous-graphe
        AtomSmile = Chem.MolToSmiles(submol, rootedAtAtom=1)           
        return AtomSmile
    
    def generate_neighbour(self, molgraph_builder, depth, evaluation_strategy, return_mol_graph=False):
        """
        :param molgraph_builder: evomol.molgraphops.molgraph.MolGraphBuilder instance previously set up to apply
        perturbations on the desired molecular graph.
        :param depth in number of perturbations of the output neighbour.
        :param evaluation_strategy: evomol.evaluation.EvaluationStrategyComposite instance that is used to evaluate the
        solutions in the EvoMol optimization procedure
        :param return_mol_graph: whether to return the molecular graph (evomol.molgraphops.molgraph.MolGraph) or a
        SMILES.
        :return: list of (evomol.molgraphops.molgraph.MolGraph, string id of the perturbation, list of chosen actions) or
        (list of SMILES, string id of the perturbation, list of chosen actions)
        """
        # Initialization of molecular graph ID
        id = _compute_root_node_id()
        # Initializing the list of molgraph_builders and chosen_actions to be updated
        molgraph_builders = []
        chosen_actions = []
        # Initializing the depth counter and the current_features list
        self.depth_counter = 0
        self.current_features = []
        # Iterating over the number of actions to be executed
        for i in range(depth):
            # Copying and sanitizing (kekulize) QuMolGraphBuilder
            molgraph_builder = molgraph_builder.copy()
            # Getting the ECFP of the current molecule
            ecfps_trace = {}
            mol = MolFromSmiles(molgraph_builder.qu_mol_graph.to_smiles())
            print ("QLearningActionSelectionStrategy - generate_neighbour - kekulised mol : ", Chem.MolToSmiles(mol, kekuleSmiles=True))
            
            ############# Generate ecfp for each part of the mol ###########
            #Management of mol graph and smiles issue if r=0
            r=int(self.ecfp / 2) 
            AllChem.GetMorganFingerprint(mol, r , bitInfo=ecfps_trace)
            
            # Keeping only contexts ecfp of required diameter
            ecfps_trace = {key: tuple(item for item in value if item[1] == int(self.ecfp / 2))
                           for key, value in ecfps_trace.items()
                           if any(item[1] == int(self.ecfp / 2) for item in value)}
            ############# END Generate ecfp for each part of the mol ###########
            
            ############# Generate tools permitting to interpret ecfp at atom level ###########
            # Putting atom ids as keys and the corresponding ECFP and radius as values
            NEWecfps = {}
            # Creating directories if they don't exist
            makedirs(dirname("./Potential_FingerPrint/"), exist_ok=True)
            for k in ecfps_trace.keys():
                #Drawing molgraph
                SubMolGraph = Draw.DrawMorganBit(mol, k, ecfps_trace)
                with open("./Potential_FingerPrint/morgan_bit"+str(k)+".svg", "w") as f:
                     f.write(SubMolGraph)
               
                for (atom_idx, rad) in ecfps_trace[k]:
                    # NEWecfps : DICTIONARY CONTAINING FOR EACH ATOM POSITION: 
                    # ecfp, position in the list of ecfp
                    position = int(np.where(self.valid_ecfps == k)[0])+1
        
                    NEWecfps.update({atom_idx: (k, position)})
            ############# END ecfp interpretation tools ###########

            ############# Selecting the action to run in the current mol  ###########
            chosen_action, chosen_action_ecfp_contexts = self.select_action_type(NEWecfps, molgraph_builder)
            # Updating molecule ID
            id = _compute_new_node_id(id, chosen_action)
            # Applying action
            molgraph_builder.execute_action_coords(chosen_action)
            # Updating the list of sanitized (kekulize) molgraph_builders and chosen_actions
            molgraph_builders.append(molgraph_builder.copy())
            chosen_actions.append(chosen_action)
            Mol_kek = Chem.MolToSmiles(MolFromSmiles(molgraph_builder.qu_mol_graph.to_smiles()), kekuleSmiles=True)
                    
            print ("QLearningActionSelectionStrategy - generate_neighbour - data return: kekukized smiles:", Mol_kek)
            
        if return_mol_graph:
            return molgraph_builder.qu_mol_graph, id, molgraph_builders, chosen_actions, chosen_action_ecfp_contexts
        else:
            return molgraph_builder.qu_mol_graph.to_aromatic_smiles(), id, molgraph_builders, chosen_actions, chosen_action_ecfp_contexts
            
    @abstractmethod 
    def initialize_weights(self, file_path, number_of_accepted_atoms):
        """
        Initializing the weights for each action type
        :param file_path: path to the file containing the initial weights
        :param number_of_accepted_atoms: number of accepted atoms in the molecule
        return: list of initial weights
        """

        pass

    @abstractmethod
    def select_action_type(self, action_types_list, evaluation_strategy, ecfps=None, NEWecfps = None, molgraph_builder=MolGraphBuilder([], [])):
        """
        Selecting the action type according to the Q-learning strategy
        :param action_types_list: list of action types authorized
        :param evaluation_strategy: evaluation strategy
        :param ecfps: list of ECFP of the current molecule
        :param molgraph_builder: molecule graph builder
        :return: valid action to execute
        """

        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Updating the weights of the Q-learning strategy
        :param args: arguments of the update method
        :param kwargs: keyword arguments of the update method
        """
        pass

class SuccessRate:
    """
    Class for storing the context success rate
    """
    def __init__(self, use_success_tuple):
        self.use = use_success_tuple[0]
        self.success = use_success_tuple[1]       
        
    def get_success_rate(self, p0=None):
        return self.success / self.use if self.use > 0 else p0 if p0 else 0.

class StochasticQLearningActionSelectionStrategy(QLearningActionSelectionStrategy):
    """
    Stochastic Q-Learning action selection strategy based on the success rate of the contexts
    """

    def __init__(self, depth, number_of_accepted_atoms, epsilon, ecfp, valid_ecfp_file_path=None,
                 init_weights_file_path=None, preselect_action_type=False, disable_updates=False,
                 epsilon_min=0.2, epsilon_0=1, lambd=0.01, alpha=0.4, step=0, epsilon_method="power_law"):
        """
        :param depth: number of consecutive executed actions before evaluation
        :param number_of_accepted_atoms: number of accepted atoms in the molecule
        :param p0: initial context probability
        :param epsilon: exploration rate
        :param valid_ecfp_file_path: path to the file containing the valid ECFPs
        :param init_weights_file_path: initial weights for the Q-learning strategy
        :param preselect_action_type: whether to preselect the action type
        before selecting the actual action
        """

        super().__init__(depth, number_of_accepted_atoms, ecfp, valid_ecfp_file_path=valid_ecfp_file_path,
                         init_weights_file_path=init_weights_file_path, preselect_action_type=preselect_action_type,
                         disable_updates=disable_updates)

        self.epsilon = epsilon
        self.AllFeatDecod = {action:{} for action in ['AddA', 'RmA', 'ChB']} 
        
        self.epsilon_min = epsilon_min
        self.epsilon_0 = epsilon_0
        self.Greedylambda = lambd
        self.PLalpha = alpha
        self.step = step
        self.epsilon_method = epsilon_method  # Add epsilon method parameter
    
    # Manage chosen epsilon method
    def get_current_epsilon(self):
        """
        Get current epsilon value based on selected method
        """
        if self.epsilon_method == "greedy":        
            return max(self.epsilon_min, self.epsilon_0 * np.exp(-self.Greedylambda * self.step))
    
        elif self.epsilon_method == "power_law":
            return max(self.epsilon_min, self.epsilon_0 / (1 + self.step)**self.PLalpha)
        else:
            return self.epsilon # Default epsilon value if no method is selected
                
    def initialize_weights(self, file_path, number_of_accepted_atoms, actions):
        """
        Initializing the weights for each action type
        :param file_path: path to the file containing the initial weights
        :param number_of_accepted_atoms: number of accepted atoms in the molecule
        return: list of initial weights
        """
        try:
            if file_path is None:
                # Returning the weights for each action type
                #Initialising Weights : dictionary with type of actions as keys and {position:(use, success, success_rate)} as values
                Weights = {action:{} for action in actions} 
                for action in actions:
                    #stocker self.Weight as a dict per action type; for each ecfp*context index, keep (use, success, success_rate)
                    Weights[action] = defaultdict(lambda: [0, 0, SuccessRate((0,0)).get_success_rate(self.epsilon)]) 
                                
            else:
                # Reading the last line of the file to get the initial weights
                with open(file_path, "r") as f:
                    init_weights = json.load(f)
                    for action in actions:
                        Weights[action].update(SuccessRate(use_success_tuple) for use_success_tuple in init_weights[action])
            return Weights 

        except FileNotFoundError:
            print('The file containing the initial weights does not exist.')

   
    def Update_Decod_Dict(self, action, curr_feat):
        """ 
        Update data_dict dictionary for the key of interest depending if already existing key or not
        params curr_data : data to add in data_dict with all steps saving for some keys
        """
        # Keep "Mol_SMILES / Pos_atom" data in a list containing values of all steps
        for key, curr_data in curr_feat.items(): #key = curr_feat.keys()
            if key not in self.AllFeatDecod[action]:
                new_entry = curr_data.copy()
                new_entry['Chosen_action'] = [curr_data['Chosen_action']]
                new_entry["Mol_SMILES"] = [curr_data["Mol_SMILES"]]
                new_entry['atom position'] = [curr_data['atom position']]
                self.AllFeatDecod[action][key] = new_entry
            else:
                self.AllFeatDecod[action][key]['Chosen_action'].append(curr_data['Chosen_action'])
                self.AllFeatDecod[action][key]["Mol_SMILES"].append(curr_data["Mol_SMILES"])
                self.AllFeatDecod[action][key]['atom position'].append(curr_data['atom position'])
        
    def select_action_type(self, NEWecfps=None, molgraph_builder=MolGraphBuilder([], [])):
        """
        Action_type_selection depending probability of each contexts
        #:param action_types_list: list of action types authorized
        #:param evaluation_strategy: evaluation strategy
        :param ecfps: list of ECFP of the current molecule
        :param molgraph_builder: graph builder of the current molecule
        :return: action selected
        """

        if NEWecfps is None:
            NEWecfps = []

        ############ Management of the action type selection (superAction of specific action) ############
        # The perturbation type is selected before selecting the actual perturbation
        if self.preselect_action_type:
            # Drawing the action type within the action spaces
            # Simpliest case: 
            actions = [np.random.choice(molgraph_builder.get_action_spaces_keys())]
        else:
            # Getting the action spaces of the current molecule
            actions = molgraph_builder.get_action_spaces_keys() # actions: ['AddA', 'RmA', 'ChB']
        ############ END Management of the action type selection (superAction of specific action) ############
        # Initializing the valid actions for each action type
        valid_action_index_list = {action: [] for action in actions} #index per action if preselect action
        valid_action_index_overall_list = []
        
        #Generating the list of valid action to perform
        for action in actions:
            # Initializing the list of valid actions for the current action type
            valid_action_coords_list = []
            # Getting the valid actions for the current action type
            action_mask = molgraph_builder.get_valid_mask_from_key(action) 
            #return the validity of the given action (True/false)
            #len(action_mask)=156 for AddA; 38 for RmA & 2812 for chB
            valid_actions = np.nonzero(action_mask)#len(valid_actions) =1 whereas several references are proposed
            #return a tupple containing an array containing the indices of True elements in action_mask
            
            # Creating a list containing the concatenation of pair ('action', valid action coordinate) 
            # for all valid actions coordinates from the valid_actions tupple
            # Utilisation de zip et d'une liste de compréhension
            for v in valid_actions[0]:
                list = (action, int(v)) 
                valid_action_coords_list.append(list)
                
            # valid_action_index_list concatenate the pair ('action j',coordonate_valid_action i) for all actions defined in this experimentation
                valid_action_index_overall_list.append(list)
                valid_action_index_list[action].append(int(v))
        
        # Random draw of a number ranged in [0;1]. If <epsilon, uniform random draw of an action, else, classic selection
        alea = random.uniform(0, 1)

        # Get current epsilon based on selected method
        curr_epsilon = self.get_current_epsilon()
                            
        if alea > curr_epsilon:
            # Computing the feature vectors for each action type
            if valid_action_index_overall_list:
                #Dictionaries initialisation
                Index_list = {action: [] for action in actions} 
                COO_Features = {action:() for action in actions} 
                Proba = {action:[] for action in actions}
                
                for valid_action in valid_action_index_overall_list:
                    action_space = molgraph_builder.action_spaces_d[valid_action[0]]
             #return the reference of the ActionSpaces object on their identifier in a dictionary attribute
            # e.g: <evomol.molgraphops.actionspace.ChangeBondActionSpace object at 0x72ff276589a0>
                    #Extracting the (ecfp*action) position + decod with ecfp, atom position in mol, type of atom, idx of action and type of action 
                    NEWaction_contexts, NewFeatDecod = self.extract_features(molgraph_builder, NEWecfps, valid_action, action_space)
                    #Index_List contains the ecfp contexts correspondign to the valid actions
                    Index_list[valid_action[0]].append(int(NEWaction_contexts))
                    
                    #Adding NewFeatDecod to FeatDecod
                    self.Update_Decod_Dict(valid_action[0], NewFeatDecod)
                    Proba[valid_action[0]].append(self.Weights[valid_action[0]][NEWaction_contexts][2])
                
                for action in actions:
                    row = np.zeros(len(Index_list[action]), dtype=int)
                    data = np.ones(len(Index_list[action]), dtype=int) 
                    COO_Features[action] = coo_matrix((data, (row,Index_list[action])))
                    COO_valid_action_features = COO_Features[action].toarray()
                    
                AllSuccessRate = [value for values in Proba.values() for value in values]
                probability_divider = sum(sum(values) for values in Proba.values())
                
        # Computing the probability distribution for each action type and choose one accordingly
                if probability_divider == 0.0: 
                    chosen_action_index = np.random.choice(len(AllSuccessRate))
                else:
                    proba = [value / probability_divider for value in AllSuccessRate]
            #### Select action using a random weighted by success rate
                    chosen_action_index = np.random.choice(len(AllSuccessRate), p=proba)
        # Getting the chosen action
                chosen_action = valid_action_index_overall_list[chosen_action_index]
                chosen_action_ecfp_contexts, chosen_action_Decod = self.extract_features(molgraph_builder, NEWecfps, chosen_action, molgraph_builder.action_spaces_d[chosen_action[0]])

        else : #if alea < self.epsilon:
             chosen_action_index = np.random.choice(np.arange(len(valid_action_index_overall_list)))
             chosen_action = valid_action_index_overall_list[chosen_action_index]
             action_space = molgraph_builder.action_spaces_d[chosen_action[0]]
             #Extract ecfp context and feature Decod of chosen action
             chosen_action_ecfp_contexts, chosen_action_Decod = self.extract_features(molgraph_builder, NEWecfps, chosen_action, action_space)
             self.Update_Decod_Dict(chosen_action[0], chosen_action_Decod)
        # Incrementing the depth counter
        self.depth_counter += 1
        return chosen_action , chosen_action_ecfp_contexts

    def update(self, *args, **kwargs):
        """
        Updates the weights according to the action(s) chosen previously
        :molgraph_builder: molecule graph builder object for the current context
        :executed_action: action executed on the molecule
        :reward: reward obtained from the evaluation of the executed action on the current molecule
        :inverted_reward: boolean indicating whether the reward should be inverted or not
        :boolean_reward: boolean indicating whether the reward should be boolean or not
        """
        # Checking if the updates are disabled
        if self.disable_updates:
            return
        # Checking the needed arguments
        try:
            action_type = args[2][0]
            successful = args[4] # mutated_total_score           
            ecfp_context_index = args[3] 
            inverted_reward = args[5]
        except IndexError:
            raise ValueError('Invalid arguments')

        if inverted_reward:
           successful = not successful
        #stocker self.Weight as a dict per action type; for each ecfp*context index, keep (use, success, success_rate)
        self.Weights[action_type][ecfp_context_index][0] += 1  # Incrémentation du nombre d'utilisations
        self.Weights[action_type][ecfp_context_index][1] += successful  # Incrémentation du nombre de succès si success=1
        self.Weights[action_type][ecfp_context_index][2] = SuccessRate((self.Weights[action_type][ecfp_context_index][0], self.Weights[action_type][ecfp_context_index][1])).get_success_rate(self.epsilon)  # Recalcule du taux de succès
        
        # Decrementing the depth counter
        self.depth_counter -= 1
        self.step += 1 

class DeterministQLearningActionSelectionStrategy(QLearningActionSelectionStrategy):
    """
    Selection of the action type according to a determinist Q-learning strategy.
    """
    def __init__(self, depth, number_of_accepted_atoms, alpha, epsilon, gamma, ecfp, valid_ecfp_file_path=None,
                 init_weights_file_path=None, preselect_action_type=False, disable_updates=False):
        """
        :param depth: number of consecutive executed actions before evaluation
        :param number_of_accepted_atoms: number of accepted atoms in the molecule
        :param alpha: learning rate
        :param epsilon: exploration rate
        :param gamma: discount factor
        :param valid_ecfp_file_path: path to the file containing the valid ECFPs
        :param init_weights_file_path: initial weights for the Q-learning strategy
        :param preselect_action_type: whether to preselect the action type
        before selecting the actual action
        """

        super().__init__(depth, number_of_accepted_atoms, ecfp, valid_ecfp_file_path=valid_ecfp_file_path,
                         init_weights_file_path=init_weights_file_path, preselect_action_type=preselect_action_type,
                         disable_updates=disable_updates)

        # Initializing the hyperparameters of the Q-learning strategy
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

    def initialize_weights(self, file_path, number_of_accepted_atoms):
        """
        Initializing the weights for each action type
        :param file_path: path to the file containing the initial weights
        :param number_of_accepted_atoms: number of accepted atoms in the molecule
        """

        try:
            if file_path is None:
                # Returning the weights for each action type
                return [np.array([np.random.normal() for _ in range((self.number_of_contexts + 1) * number_of_accepted_atoms)]),
                        np.array([np.random.normal() for _ in range(self.number_of_contexts + 1)]),
                        np.array([np.random.normal() for _ in range((self.number_of_contexts + 1) * 4)])]
            else:
                # Reading the last line of the file to get the initial weights
                with open(file_path, "r") as f:
                    init_weights = json.load(f)

                # Returning the weights for each action type
                return [np.array(init_weights['w_addA']),
                        np.array(init_weights['w_rmA']),
                        np.array(init_weights['w_chB'])]

        except FileNotFoundError:
            print('The file containing the initial weights does not exist.')

    def select_action_type(self, action_types_list, evaluation_strategy, ecfps=None, molgraph_builder=MolGraphBuilder([], [])):
        """
        :param action_types_list: list of action types authorized
        :param evaluation_strategy: evaluation strategy
        :param ecfps: list of ECFP of the current molecule
        :param molgraph_builder: graph builder of the current molecule
        :return: action selected
        """

        if ecfps is None:
            ecfps = []

        # Getting the action spaces of the current molecule
        actions = molgraph_builder.get_action_spaces_keys()

        # Initializing the feature vectors for each action type
        features_addA = np.array([])
        features_rmA = np.array([])
        features_chB = np.array([])

        # Initializing the valid actions for each action type
        valid_action_index_list = []

        for action in actions:
            # Initializing the list of valid actions for the current action type
            valid_action_coords_list = []

            # Getting the valid actions for the current action type
            action_mask = molgraph_builder.get_valid_mask_from_key(action)
            action_space = molgraph_builder.action_spaces_d[action]
            valid_actions = np.nonzero(action_mask)

            # Getting the valid actions coordinates
            for valid_act in valid_actions[0]:
                valid_action_coords_list.append((action, int(valid_act)))

            # Adding the valid actions coordinates to the list of valid actions
            valid_action_index_list.append(valid_action_coords_list)

            # Computing the feature vectors for each action type
            if valid_action_coords_list:
                for valid_action in valid_action_coords_list:
                    valid_action_features = self.extract_features(molgraph_builder, ecfps, valid_action, action_space)

                    if valid_action[0] == 'AddA':
                        features_addA = np.append(features_addA, valid_action_features)
                    elif valid_action[0] == 'RmA':
                        features_rmA = np.append(features_rmA, valid_action_features)
                    elif valid_action[0] == 'ChB':
                        features_chB = np.append(features_chB, valid_action_features)
                    else:
                        raise ValueError('Invalid action')

        # Reshaping the feature vectors
        features_addA = features_addA.reshape(-1, (self.number_of_contexts + 1) * len(molgraph_builder.parameters.accepted_atoms))
        features_rmA = features_rmA.reshape(-1, self.number_of_contexts + 1)
        features_chB = features_chB.reshape(-1, (self.number_of_contexts + 1) * 4)

        # Computing the Q-values for each action type
        q_states_addA = (features_addA @ self.w_addA).flatten()
        q_states_rmA = (features_rmA @ self.w_rmA).flatten()
        q_states_chB = (features_chB @ self.w_chB).flatten()

        # Choosing a random feature vector in the list of valid actions
        if random.random() < self.epsilon:
            # Initializing the list of non-empty action lists
            non_empty_action_lists = []

            # Appending the non-empty action lists
            if q_states_addA.any():
                non_empty_action_lists.append(0)
            if q_states_rmA.any():
                non_empty_action_lists.append(1)
            if q_states_chB.any():
                non_empty_action_lists.append(2)

            # Choosing a random action type
            action_index = random.choice(non_empty_action_lists)

            # Choosing a random action in the chosen action type and
            # saving the corresponding feature vector
            if action_index == 0:
                coord_index = random.randrange(0, features_addA.shape[0])
                self.current_features.append(features_addA[coord_index])
            elif action_index == 1:
                coord_index = random.randrange(0, features_rmA.shape[0])
                self.current_features.append(features_rmA[coord_index])
            elif action_index == 2:
                coord_index = random.randrange(0, features_chB.shape[0])
                self.current_features.append(features_chB[coord_index])
            else:
                raise ValueError('Invalid action')

            # Incrementing the depth counter
            self.depth_counter += 1

            return valid_action_index_list[action_index][coord_index]

        # Choosing the best feature vector in the list of valid actions
        else:
            # Initializing the list of non-empty action lists
            non_empty_action_list_indexes = dict()

            # Appending the non-empty action lists
            if q_states_addA.any():
                non_empty_action_list_indexes['0'] = np.argmax(q_states_addA)
            if q_states_rmA.any():
                non_empty_action_list_indexes['1'] = np.argmax(q_states_rmA)
            if q_states_chB.any():
                non_empty_action_list_indexes['2'] = np.argmax(q_states_chB)

            if not non_empty_action_list_indexes:
                raise ValueError('No valid actions')

            # Choosing the best action type key
            q_max_keys = max(non_empty_action_list_indexes, key=lambda x: non_empty_action_list_indexes[x])

            # Choosing the best action in the chosen action type and
            # saving the corresponding feature vector
            if q_max_keys == '0':
                self.current_features.append(features_addA[non_empty_action_list_indexes[q_max_keys]])
                valid_action_index = non_empty_action_list_indexes[q_max_keys]
            elif q_max_keys == '1':
                self.current_features.append(features_rmA[non_empty_action_list_indexes[q_max_keys]])
                valid_action_index = non_empty_action_list_indexes[q_max_keys]
            elif q_max_keys == '2':
                self.current_features.append(features_chB[non_empty_action_list_indexes[q_max_keys]])
                valid_action_index = non_empty_action_list_indexes[q_max_keys]
            else:
                raise ValueError('Invalid action')

            # Incrementing the depth counter
            self.depth_counter += 1

            return valid_action_index_list[int(q_max_keys)][valid_action_index]

    def update(self, *args, **kwargs):
        #Called in notify_observers method (notification.py) called in mutate method defined in mutation.py
        """
        Updates the weights according to the action(s) chosen previously
        :molgraph_builder: molecule graph builder object for the current context
        :executed_action: action executed on the molecule
        :reward: reward obtained from the evaluation of the executed action on the current molecule
        :inverted_reward: boolean indicating whether the reward should be inverted or not
        :boolean_reward: boolean indicating whether the reward should be boolean or not
        """

        # Checking if the updates are disabled
        if self.disable_updates:
            return

        # Checking the needed arguments
        try:
            molgraph_builder = args[1]
            executed_action = args[2]
            action_type = args[2][0]
            context_index = args[2][1]
            reward = args[3]
            inverted_reward = args[4]
            boolean_reward = args[5]
        except IndexError:
            raise ValueError('Invalid arguments')

        # Checking the optional arguments
        if inverted_reward:
            if boolean_reward:
                if reward == 0.:
                    reward = 1.
                else:
                    reward = 0.
            else:
                reward = -reward

        # Initializing the feature vector and the matrix of valid actions
        features = np.array([])
        valid_action_coords_list = []

        # Getting the valid actions for the current context
        action_mask = molgraph_builder.get_valid_mask_from_key(executed_action[0])
        action_space = molgraph_builder.action_spaces_d[executed_action[0]]
        valid_actions = np.nonzero(action_mask)

        # Getting the ECFP of the current molecule
        ecfps_trace = {}
        AllChem.GetMorganFingerprint(MolFromSmiles(molgraph_builder.qu_mol_graph.to_smiles()), 0, bitInfo=ecfps_trace)

        # Putting atom ids as keys and the corresponding ECFP and radius as values
        ecfps = dict()

        for k in ecfps_trace.keys():
            for (atom_id, rad) in ecfps_trace[k]:
                if not atom_id in ecfps.keys():
                    ecfps[atom_id] = list()
                ecfps[atom_id].append(k)

        # Getting the features of the current context
        current_features = self.current_features[self.depth - self.depth_counter]

        # Getting the valid actions coordinates
        for valid_act in valid_actions[0]:
            valid_action_coords_list.append((executed_action[0], int(valid_act)))

        # Extracting the features of the valid actions
        if valid_action_coords_list:
            for executed_action in valid_action_coords_list:
                valid_action_features = self.extract_features(molgraph_builder, ecfps, executed_action, action_space)
                features = np.append(features, [valid_action_features])

        # Updating the weights according to the executed action
        if executed_action[0] == 'AddA':
            features = features.reshape(-1, (self.number_of_contexts + 1) * len(molgraph_builder.parameters.accepted_atoms))
            q_states = features @ self.w_addA

            target = reward + self.gamma * np.amax(q_states)
            q_state = current_features @ self.w_addA.reshape(-1, 1)

            self.w_addA = np.array([self.w_addA[i] - 2 * self.alpha * current_features[i] * (q_state - target) for i in range(len(self.w_addA))])

            # Decrementing the depth counter
            self.depth_counter -= 1
        elif executed_action[0] == 'RmA':
            features = features.reshape(-1, self.number_of_contexts + 1)
            q_states = features @ self.w_rmA

            target = reward + self.gamma * np.amax(q_states)
            q_state = current_features @ self.w_rmA.reshape(-1, 1)

            self.w_rmA = np.array([self.w_rmA[i] - 2 * self.alpha * current_features[i] * (q_state - target) for i in range(len(self.w_rmA))])
            self.depth_counter -= 1
        elif executed_action[0] == 'ChB':
            features = features.reshape(-1, (self.number_of_contexts + 1) * 4)
            q_states = features @ self.w_chB

            target = reward + self.gamma * np.amax(q_states)
            q_state = current_features @ self.w_chB.reshape(-1, 1)

            self.w_chB = np.array([self.w_chB[i] - 2 * self.alpha * current_features[i] * (q_state - target) for i in range(len(self.w_chB))])
            self.depth_counter -= 1
        else:
            raise ValueError('Invalid action')
