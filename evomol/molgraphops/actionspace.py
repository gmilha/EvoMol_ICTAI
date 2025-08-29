from abc import ABC, abstractmethod

import numpy as np
import networkx as nx
from networkx import connected_components, node_connected_component


class ActionSpace(ABC):
    """
    Abstract base class of all the Action Spaces classes.
    The subclasses must implement methods that define the space of their actions : they must provide a mask of valid
    actions, give the size of their action space, execute any action of their action space and provide a string
    description of any action of their action space.
    The action space must be linear and each action must have a integer id.
    ActionSpace subclasses are identified by a unique name that they must define
    The size of the action space must be immutable for each instance of the subclasses. Though, it can vary for a given
    subclass depending on the value of the parameters of the constructor.
    """

    class ActionSpaceParameters:
        """
        Class containing all the parameters that can be needed by the ActionSpace subclasses
        """

        def __init__(self, max_heavy_atoms=None, accepted_atoms=None, accepted_structures=None,
                     accepted_substitutions=None):
            self.max_heavy_atoms = max_heavy_atoms
            self.accepted_atoms = accepted_atoms
            self.accepted_structures = accepted_structures
            self.accepted_substitutions = accepted_substitutions

    def __init__(self, check_validity=True):
        self.check_validity = check_validity

    @abstractmethod
    def action_space_type_id(self):
        """
        Returning the id of the ActionSpace subclass)
        :return:
        """
        pass

    @abstractmethod
    def get_valid_actions_mask(self, parameters, qu_mol_graph):
        """
        Returning a boolean vector representing the validity of any action of the action space.
        :return: boolean vector of the size of the action space
        """
       # print("actionspace.py - class ActionSpace - method get_valid_actions_mask")
        pass

    @abstractmethod
    def get_action_space_size(self, parameters, qu_mol_graph):
        """
        Returning a numerical value representing the size of the action space.
        :return: numerical value
        """
        pass

    @abstractmethod
    def get_action_expl(self, id_action, parameters, qu_mol_graph):
        """
        Returning a short string description of the action of given id
        Warning : this method is only implemented for valid actions
        @param parameters:
        @param id_action:
        @param qu_mol_graph:
        @return: string list
        """

    @abstractmethod
    def get_context(self, action_id, parameters, *args, **kwargs):
        """
        Gives the positions of the atoms that were modified
        """
        pass

    @abstractmethod
    def execute_action(self, action_id, parameters, qu_mol_graph):
        """
        Executing the action identified by the given action id to the given molecular graph
        :return: boolean value representing whether the action is terminal
        """
        
      #  print("actionspace.py - class ActionSpace - execute_action method")
        validity = self.check_validity
        #print("self.check_validity :", validity)
        #print("parameters :", parameters)
        #print("qu_mol_graph :", qu_mol_graph)
        #print("action_id :", action_id)
        valid_act_mask = self.get_valid_actions_mask(parameters, qu_mol_graph)[action_id]
        #print("self.get_valid_actions_mask(parameters, qu_mol_graph)[action_id] :", valid_act_mask )
        if validity and not valid_act_mask :
        #if self.check_validity and not self.get_valid_actions_mask(parameters, qu_mol_graph)[action_id]:
            raise Exception("Trying to apply invalid action : " + self.action_to_str(action_id, parameters,
                                                                                     qu_mol_graph))

    @abstractmethod
    def action_to_str(self, action_id, parameters, qu_mol_graph):
        """
        Describing in natural language the action of the given action identifier
        :return:
        """
        #print("actionspace.py - class ActionSpace - method action_to_str")
        pass


class CutAtomV2ActionSpace(ActionSpace):
    """
    Defining the action space for atom cut : removing an atom having exactly two bonds of types t1 and t2 and no other
    bond and creating a single bond between the two atoms it was bonded to that do not share a bond
    """

    def __init__(self, check_validity=True):
        super().__init__(check_validity=check_validity)

    def action_space_type_id(self):
        return "CutA"

    def get_valid_actions_mask(self, parameters, qu_mol_graph):
        #print("actionspace.py - class CutAtomV2ActionSpace - method get_valid_actions_mask")
        # Action space validity initialization
        action_space = np.full((parameters.max_heavy_atoms,), False)

        formal_charge_vector = qu_mol_graph.get_formal_charge_vector()

        # Action only possible if the current atom is mutable and has exactly two simple bonds to two unconnected to
        # each other atoms that have no formal charges
        for i in range(qu_mol_graph.get_n_atoms()):
            bonds_count = 0
            bonds_to = []

            for j in range(qu_mol_graph.get_n_atoms()):
                bond_type_num = qu_mol_graph.get_bond_type_num(i, j)

                if bond_type_num > 0:
                    bonds_count += 1
                    bonds_to.append(j)

            if bonds_count == 2:
                # Action if the two remaining atoms do not share a bond and have no formal charges
                action_space[i] = qu_mol_graph.get_bond_type_num(bonds_to[0], bonds_to[1]) == 0 and \
                                  formal_charge_vector[0] == 0 and formal_charge_vector[1] == 0 and \
                                  qu_mol_graph.get_atom_mutability(i)

        return action_space

    def get_action_expl(self, action_id, parameters, qu_mol_graph):

        return qu_mol_graph.get_atom_type(action_id)

    def get_action_space_size(self, parameters, qu_mol_graph):
        return parameters.max_heavy_atoms

    def get_context(self, action_id, parameters, *args, **kwargs):
        raise Exception("Method not implemented yet")

    def execute_action(self, action_id, parameters, qu_mol_graph):
        super(CutAtomV2ActionSpace, self).execute_action(action_id, parameters, qu_mol_graph)
      #  print("actionspace.py - class CutAtomV2ActionSpace - method execute_action")

        #print("parameters :", parameters)
        #print("qu_mol_graph :", qu_mol_graph)
        #print("action_id :", action_id)
       
        # Extracting neigbours of atom to be cut
        neighbours = np.argwhere(qu_mol_graph.get_adjacency_matrix()[action_id] == 1).flatten()
        #print("neighbours :", neighbours)

        i, j = tuple(list(neighbours))

        # Removing bonds
        qu_mol_graph.set_bond(action_id, int(i), 0, update_repr=False)
        qu_mol_graph.set_bond(action_id, int(j), 0, update_repr=False)

        # Creating bond
        qu_mol_graph.set_bond(i, j, 1, update_repr=False)

        try:
            # Removing cut atom
            qu_mol_graph.rm_atom(action_id)

        except Exception as e:
            print("CutV2 caused error")
            raise e

    def action_to_str(self, action_id, parameters, qu_mol_graph):

        # Extracting neigbours of atom to be cut
        neighbours = np.argwhere(qu_mol_graph.get_adjacency_matrix()[action_id] == 1)

        i, j = tuple(list(neighbours))

        return "Cutting atom " + str(action_id) + " of type " + qu_mol_graph.get_atom_type(action_id) + \
               " bond to atom " + str(i) + " of type " + qu_mol_graph.get_atom_type(i) + " and atom " + str(j) + \
               " of type " + qu_mol_graph.get_atom_type(j)


class InsertCarbonAtomV2ActionSpace(ActionSpace):
    """
    Defining the action space for atom insertion between two existing atoms that have no formal charges in the molecular
    graph. The inserted atom is linked with two single bonds to the two existing atoms. The initial bond is removed.
    """

    def __init__(self, check_validity=True):
        super().__init__(check_validity=check_validity)

    def action_space_type_id(self):
        return "InsA"

    def get_valid_actions_mask(self, parameters, qu_mol_graph):
       # print("actionspace.py - class InsertCarbonAtomV2ActionSpace - method get_valid_actions_mask")

        # Action space validity initialization
        action_space = np.full((parameters.max_heavy_atoms, parameters.max_heavy_atoms), False)

        formal_charge_vector = qu_mol_graph.get_formal_charge_vector()

        # Insertion only possible if the molecule is not at its max size
        if qu_mol_graph.get_n_atoms() < parameters.max_heavy_atoms:

            # Computing the upper triangular matrix : insert on existing bonds of non charged atoms if at least one atom
            # is mutable
            for i in range(qu_mol_graph.get_n_atoms()):
                for j in range(i + 1, qu_mol_graph.get_n_atoms()):
                    action_space[i][j] = qu_mol_graph.get_bond_type_num(i, j) > 0 and formal_charge_vector[i] == 0 and \
                                         formal_charge_vector[j] == 0 and (qu_mol_graph.get_atom_mutability(i) or
                                                                           qu_mol_graph.get_atom_mutability(j))

        # Returning the values of the upper triangular matrix
        return action_space[np.triu_indices(parameters.max_heavy_atoms, k=1)]

    def _action_id_to_atoms_idx(self, action_id, parameters):
        """
        Converting the id of action to the indices of both atoms involved
        @return:
        """
        return np.triu_indices(parameters.max_heavy_atoms, k=1)[0][action_id], \
               np.triu_indices(parameters.max_heavy_atoms, k=1)[1][action_id]

    def get_action_expl(self, action_id, parameters, qu_mol_graph):

        # Computing indices of both atoms involved in the action
        at1_idx, at2_idx = self._action_id_to_atoms_idx(action_id, parameters)

        # Extracting types of the couple of atoms
        first_at_type = qu_mol_graph.get_atom_type(at1_idx)
        second_at_type = qu_mol_graph.get_atom_type(at2_idx)

        return first_at_type + ":" + second_at_type

    def get_action_space_size(self, parameters, qu_mol_graph):

        # The action space is of size sum from 1 to max_heavy_atoms -1
        return parameters.max_heavy_atoms * (parameters.max_heavy_atoms - 1) // 2

    def get_context(self, action_id, parameters, *args, **kwargs):
        return self._action_id_to_atoms_idx(action_id, parameters)

    def execute_action(self, action_id, parameters, qu_mol_graph):
        super(InsertCarbonAtomV2ActionSpace, self).execute_action(action_id, parameters, qu_mol_graph)
        #print("actionspace.py - class InsertCarbonAtomV2ActionSpace - method execute_action")
        #print("parameters :", parameters)
        #print("qu_mol_graph :", qu_mol_graph)
        #print("action_id :", action_id)
        # Mapping the action id on a complete matrix
        matrix_id = \
            np.arange(parameters.max_heavy_atoms ** 2).reshape(parameters.max_heavy_atoms, parameters.max_heavy_atoms)[
                np.triu_indices(parameters.max_heavy_atoms, k=1)][action_id]

        # Extracting the coordinates on the complete matrix of the given action id
        i = int(matrix_id % parameters.max_heavy_atoms)
        j = int(matrix_id // parameters.max_heavy_atoms)

        # Removing bond between atoms
        qu_mol_graph.set_bond(i, j, 0, update_repr=False)

        # Adding the atom
        qu_mol_graph.add_atom("C")

        # Creating a bond from the last inserted atom to the existing ones
        qu_mol_graph.set_bond(qu_mol_graph.get_n_atoms() - 1, i, 1, update_repr=False)
        qu_mol_graph.set_bond(qu_mol_graph.get_n_atoms() - 1, j, 1, update_repr=False)

        try:
            # Updating the molecular graph after addition
            qu_mol_graph.end_atom_addition_procedure()

        except Exception as e:
            print("Insert V2 caused error")
            raise e

        # Non terminal action
        return True

    def action_to_str(self, action_id, parameters, qu_mol_graph):

        # Mapping the action id on a complete matrix
        matrix_id = np.arange(parameters.max_heavy_atoms ** 2).reshape(parameters.max_heavy_atoms,
                                                                       parameters.max_heavy_atoms)[
            np.triu_indices(parameters.max_heavy_atoms, k=1)][action_id]

        # Extracting the coordinates on the complete matrix of the given action id
        i = int(matrix_id % parameters.max_heavy_atoms)
        j = int(matrix_id // parameters.max_heavy_atoms)

        return "Insert simple bond between atoms of ids " + str(i) + " (" + qu_mol_graph.get_atom_type(
            i) + ")" + " and " + str(j) + " (" + qu_mol_graph.get_atom_type(j) + ")"


class AddAtomActionSpace(ActionSpace):
    """
    Defining the action space for atom addition into the molecular graph.
    An atom of accepted type can be inserted into the molecular graph if the maximum number of heavy atoms has not
    been reached.
    With the allow_bonding option set to True, atoms can be inserted without bonding or with bonding to an atom that
    has free electrons and that is connectable.
    With keep_connected option set to True, an atom can be only inserted without connexion if the molecular graph is
    empty.
    """

    def __init__(self, allow_bonding=False, keep_connected=False, check_validity=True):
        """
        Recording user options. If keep_connected is set to True, it overrides the value of allow_bonding.
        :param allow_bonding: whether atoms can be inserted with a bond to other atoms
        :param keep_connected: whether the connectivity of the graph is guaranteed : atoms can be inserted without bond
        iff. the molecular graph is empty. Otherwise they can only be inserted with a bond to an other existing atom.
        """
        super().__init__(check_validity=check_validity)

        # Overriding the value of allow_bonding if keep_connected is set to True
        if keep_connected:
            self.keep_connected = True
            self.allow_bonding = True
        else:
            self.keep_connected = keep_connected
            self.allow_bonding = allow_bonding

    def action_space_type_id(self):
        return "AddA"

    def get_action_expl(self, action_id, parameters, qu_mol_graph):

        # Converting action id to coordinates in the action space
        i = action_id // len(parameters.accepted_atoms)
        j = action_id % len(parameters.accepted_atoms)

        # Computing atom type that will be added on current action
        new_at_type = parameters.accepted_atoms[j]

        # Old atom type initialization
        old_at_type = ""

        # Computing atom type of the atom on which the new atom will be bonded (if it is defined)
        if i >= 1:
            old_at_id = i - 1
            old_at_type = qu_mol_graph.get_atom_type(old_at_id)

        return old_at_type + ":" + new_at_type

    def get_valid_actions_mask(self, parameters, qu_mol_graph):
       # print("actionspace.py - class AddAtomActionSpace - method get_valid_actions_mask")

        if self.allow_bonding:

            # Action space validity initialization
            action_space = np.full((parameters.max_heavy_atoms + 1, len(parameters.accepted_atoms)), False)

            # Computing the possibility of adding an atom without connexion
            if self.keep_connected:
                # Possibility of adding an unconnected atom iff the molecular graph is empty
                action_space[0] = np.full((len(parameters.accepted_atoms),), qu_mol_graph.get_n_atoms() == 0)
            else:
                # Possibility of adding an unconnected atom iff the molecular graph is not full
                action_space[0] = np.full((len(parameters.accepted_atoms),),
                                          qu_mol_graph.get_n_atoms() < parameters.max_heavy_atoms)

            # Computing the possibility of adding an atom with a connexion
            if qu_mol_graph.get_n_atoms() < parameters.max_heavy_atoms:

                # Extracting the free electrons vector of the molecular graph
                free_electons_vect = qu_mol_graph.get_free_electrons_vector()

                # Extracting the formal charges vector
                formal_charges_vect = qu_mol_graph.get_formal_charge_vector()

                # Iterating over the list of defined atoms in the molecular graph
                for i in range(0, qu_mol_graph.get_n_atoms()):
                    # The new atom can be bonded to the current atom if the current atom has free electrons left and if
                    # it has no formal charge
                    action_space[i + 1] = np.full((len(parameters.accepted_atoms),),
                                                  free_electons_vect[i] >= 1 and formal_charges_vect[i] == 0)

        else:
            # All atoms types are insertable iff. the current size of the molecular graph has not reached the limit
            action_space = np.repeat(qu_mol_graph.get_n_atoms() < parameters.max_heavy_atoms,
                                     len(parameters.accepted_atoms))

        return action_space.reshape(-1, )

    def get_action_space_size(self, parameters, qu_mol_graph):

        if self.allow_bonding:
            # The size of the action space is equal to the cardinality of the set of insertable atom types times the
            # max size of the molecule for atom addition with bond, plus the cardinality of the set of insertable atom
            # types for simple atom addition
            return len(parameters.accepted_atoms) * (parameters.max_heavy_atoms + 1)
        else:
            # The size of the action space is equal to the cardinality of the set of insertable atom types
            return len(parameters.accepted_atoms)

    def get_context(self, action_id, parameters, *args, **kwargs):
        # Converting action id to coordinates in the action space
        number_accepted_atoms = len(parameters.accepted_atoms)

        i = action_id // number_accepted_atoms 
        #position of the atom where an addition will be run corresponds to the interger part of the action_id divided by nb of accepted_atom
        j = action_id % number_accepted_atoms
        #Type of atom to be added corresponds to the modulo of the action_id divided by nb of accepted_atom

        # Computing atom position on which the new atom will be bonded (if it is defined)
        support_atom = i - 1 if i >= 1 else i

        return [support_atom, j]

    def execute_action(self, action_id, parameters, qu_mol_graph):
        super(AddAtomActionSpace, self).execute_action(action_id, parameters, qu_mol_graph)
      #  print("actionspace.py - class AddAtomActionSpace - method execute_action")
        #print("parameters :", parameters)
        #print("qu_mol_graph :", qu_mol_graph)
        #print("action_id :", action_id)
        #print("len(parameters.accepted_atoms) :", len(parameters.accepted_atoms))
        
        # Simple addition of the atom
        if action_id < len(parameters.accepted_atoms):
           # print("action_id < len(parameters.accepted_atoms)")
            qu_mol_graph.add_atom(parameters.accepted_atoms[action_id])
           # print("method execute_action - AddA ", parameters.accepted_atoms[action_id])

        # Adding the new atom with a bond to an existing one
        else:
           # print("action_id >= len(parameters.accepted_atoms)")
            atom_type = parameters.accepted_atoms[action_id % len(parameters.accepted_atoms)]
            to_atom_id = action_id // len(parameters.accepted_atoms) - 1
           # print("method execute_action - AddA ", atom_type, "to_atom #", to_atom_id)

            # Adding the atom
            qu_mol_graph.add_atom(atom_type)

            # Creating a bond from the last inserted atom to the existing one
            qu_mol_graph.add_bond(qu_mol_graph.get_n_atoms() - 1, to_atom_id)

        try:

            # Updating the molecular graph after addition
            qu_mol_graph.end_atom_addition_procedure()

        except Exception as e:
            print("Add atom caused error")
            raise e

        # Non terminal action
        return True

    def action_to_str(self, action_id, parameters, qu_mol_graph):

        dscr = "Add atom of type " + str(parameters.accepted_atoms[action_id % len(parameters.accepted_atoms)])
        if self.allow_bonding and action_id >= len(parameters.accepted_atoms):
            dscr += " to atom of id " + str(action_id // len(parameters.accepted_atoms) - 1) + " (" + \
                    str(qu_mol_graph.get_atom_type(action_id // len(parameters.accepted_atoms) - 1)) + ")"
            print ("action performed:", dscr)
        return dscr


class RemoveAtomActionSpace(ActionSpace):
    """
    Defining the action space for atom removal from the molecular graph.
    If the keep_connected constraint is set to True, an atom can be removed only if the removal does not create two
    connected components in the molecular graph ("breaking" the molecule in half).
    If not, all the defined atom can be removed.
    Regardless of the value of keep_connected, only the mutable atoms are removable.
    """

    def __init__(self, check_validity=True, keep_connected=False):
        super().__init__(check_validity)
        self.keep_connected = keep_connected

    def action_space_type_id(self):
        return "RmA"

    def get_action_expl(self, action_id, parameters, qu_mol_graph):

        at_to_remove_id = action_id
        return qu_mol_graph.get_atom_type(at_to_remove_id)

    def get_valid_actions_mask(self, parameters, qu_mol_graph):
       # print("actionspace.py - class RemoveAtomActionSpace - method get_valid_actions_mask")
        # Extracting the list of existing atoms
        rm_atom_space_mask = np.full((parameters.max_heavy_atoms,), False)

        # If keep connected is set to True, impossible to remove the atoms whose removal would create multiple connected
        # components in the molecular graph
        if self.keep_connected:

            # Extracting the articulation points vector
            articulation_points_vector = qu_mol_graph.get_articulation_points_vector()

            # Any atom can be removed if it is mutable and (if it has only one neighbour or if none of its bonds are
            # bridges)
            for i in range(qu_mol_graph.get_n_atoms()):
                rm_atom_space_mask[i] = not articulation_points_vector[i] and qu_mol_graph.get_atom_mutability(i)

        else:
            # If keep_connected is set to False, all mutable atoms are removable
            for i in range(qu_mol_graph.get_n_atoms()):
                rm_atom_space_mask[i] = qu_mol_graph.get_atom_mutability(i)

        return rm_atom_space_mask

    def get_action_space_size(self, parameters, qu_mol_graph):
        # The number of removable atoms is the number of definable atoms
        return parameters.max_heavy_atoms

    def get_context(self, action_id, parameters, *args, **kwargs):
        #print("get_context [action_id] :",[action_id])
        return [action_id] #context for rmA corresponds to the position where the current atom could be removed 

    def execute_action(self, action_id, parameters, qu_mol_graph):
        super(RemoveAtomActionSpace, self).execute_action(action_id, parameters, qu_mol_graph)
        # print("actionspace.py - class RemoveAtomActionSpace - method execute_action")
        #print("parameters :", parameters)
        #print("qu_mol_graph :", qu_mol_graph)
        #print("action_id :", action_id)
        try:
            # Removing the atom
            qu_mol_graph.rm_atom(int(action_id))
         #   print("atom # ", int(action_id) ,"removed")

        except Exception as e:
            print("Remove atom caused error")
            raise e

        # Non terminal action
        return True

    def action_to_str(self, action_id, parameters, qu_mol_graph):
        dscr = "Remove atom of id " + str(action_id) + " (" + qu_mol_graph.get_atom_type(action_id) + ")"
       # print ("action performed:", dscr)
        return dscr


class MoveFunctionalGroupActionSpace(ActionSpace):
    """
    Definition of the action space for moving a functional group at a different place of the molecular graph
    """

    def __init__(self, check_validity=True):
        super().__init__(check_validity)

    def action_space_type_id(self):
        return "MoveFG"

    def _action_id_to_atoms_idx(self, action_id, parameters):
        """
        Converting the id of action to the indices of both atoms involved
        @return:
        """
        return np.triu_indices(parameters.max_heavy_atoms, k=1)[0][action_id // parameters.max_heavy_atoms], \
               np.triu_indices(parameters.max_heavy_atoms, k=1)[1][action_id // parameters.max_heavy_atoms]

    def get_valid_actions_mask(self, parameters, qu_mol_graph):
       # print("actionspace.py - class MoveFunctionalGroupActionSpace - method get_valid_actions_mask")

        # Initialization of the action space
        valid_action_space = np.full((parameters.max_heavy_atoms, parameters.max_heavy_atoms,
                                      parameters.max_heavy_atoms), False)

        # Extraction of the bridge matrix
        bridge_bond_matrix = qu_mol_graph.get_bridge_bonds_matrix()

        # Iterating over all bonds
        for i in range(qu_mol_graph.get_n_atoms()):
            for j in range(i + 1, qu_mol_graph.get_n_atoms()):

                # Functional group can be moved only if the bond is a bridge and none of the atoms has a formal charge
                # and at least one atom is mutable
                if bridge_bond_matrix[i][j] and qu_mol_graph.get_formal_charge(i) == 0 and \
                        qu_mol_graph.get_formal_charge(j) == 0 \
                        and (qu_mol_graph.get_atom_mutability(i) or qu_mol_graph.get_atom_mutability(j)):

                    # Extracting the current bond type and the free electrons vector
                    bond_type_num = qu_mol_graph.get_bond_type_num(i, j)
                    free_electrons_vector = qu_mol_graph.get_free_electrons_vector()

                    # Iterating over all other atoms
                    for k in range(qu_mol_graph.get_n_atoms()):
                        if k != i and k != j:
                            # The functional group can be moved if the current atom has enough electrons left and has no
                            # formal charge
                            valid_action_space[i][j][k] = free_electrons_vector[k] >= bond_type_num and \
                                                          qu_mol_graph.get_formal_charge(k) == 0

        return valid_action_space[np.triu_indices(parameters.max_heavy_atoms, k=1)].flatten()

    def get_action_space_size(self, parameters, qu_mol_graph):
        """
        The action space size is the maximum number of bonds multiplied by the maximum number of atoms
        """
        return (parameters.max_heavy_atoms * (parameters.max_heavy_atoms - 1) // 2) * parameters.max_heavy_atoms

    def get_action_expl(self, id_action, parameters, qu_mol_graph):
        return ""

    def get_context(self, action_id, parameters, *args, **kwargs):
        raise Exception("Method not implemented yet")

    def execute_action(self, action_id, parameters, qu_mol_graph):
       # print("actionspace.py - class MoveFunctionalGroupActionSpace - method execute_action")
        #print("parameters :", parameters)
        #print("qu_mol_graph :", qu_mol_graph)
        #print("action_id :", action_id)
        smiles_before = qu_mol_graph.to_aromatic_smiles()

        i, j = self._action_id_to_atoms_idx(action_id, parameters)

        # Extracting the atom on which the group is to be moved
        k = action_id % parameters.max_heavy_atoms

        # Computing imputed adjacency matrix
        imput_adjacency_matrix = qu_mol_graph.get_adjacency_matrix()
        imput_adjacency_matrix[i][j] = False
        imput_adjacency_matrix[j][i] = False

        # Extracting the connected components with the bond removed
        conn_components = list(connected_components(nx.from_numpy_array(imput_adjacency_matrix)))

        # Selecting the atom from the initial bond to be bonded to k
        if i in conn_components[0]:
            if k in conn_components[0]:
                at_to_be_bonded = j
            else:
                at_to_be_bonded = i
        else:
            if k in conn_components[0]:
                at_to_be_bonded = i
            else:
                at_to_be_bonded = j

        # Extracting the bond type to be removed
        bond_type = qu_mol_graph.get_bond_type_num(i, j)

        # Removing the bond
        qu_mol_graph.set_bond(i, j, 0, update_repr=False)

        try:

            # Setting the new bond
            qu_mol_graph.set_bond(at_to_be_bonded, k, bond_type, update_repr=True)

        except Exception as e:
            print("Move group caused error")
            print("Smiles before : " + str(smiles_before))
            print("Smiles after : " + str(qu_mol_graph.to_aromatic_smiles()))
            raise e

    def action_to_str(self, action_id, parameters, qu_mol_graph):
        return ""


class RemoveGroupActionSpace(ActionSpace):
    """
    Definition of the action space for removing a group of connected atoms from the molecular graph
    """

    def __init__(self, check_validity=True, only_remove_smallest_group=False):
        """
        :param check_validity:
        :param only_remove_smallest_group: whether both parts of bridge bonds can be removed (False), or only the
        smallest part (True)
        """
        super().__init__(check_validity)
        self.only_remove_smallest_group = only_remove_smallest_group

    def action_space_type_id(self):
        return "RemoveFG"

    def _action_id_to_atoms_idx(self, action_id, parameters):
        """
        Converting the id of action to the indices of both atoms involved
        @return:
        """
        return action_id // parameters.max_heavy_atoms, action_id % parameters.max_heavy_atoms

    def _get_connected_component_after_removing_bond(self, qu_mol_graph, bond_at_1, bond_at_2):
        """
        Returning the connected component that contains that atom of index bond_at_1 if the bond between bond_at_1 and
        bond_at_2 was removed. The list of indices of the connected component is returned
        :param qu_mol_graph:
        :param bond_at_1:
        :param bond_at_2:
        :return:
        """

        # Computing the adjacency matrix
        adjacency_matrix = qu_mol_graph.get_adjacency_matrix()

        # Virtually removing the bond in the adjacency matrix
        adjacency_matrix[bond_at_1][bond_at_2] = 0
        adjacency_matrix[bond_at_2][bond_at_1] = 0

        # Loading the graph as a networkx Graph
        g = nx.from_numpy_array(adjacency_matrix)

        # Returning the list of indices of the connected component that contains the bond_at_1 atom
        return list(node_connected_component(g, bond_at_1))

    def get_valid_actions_mask(self, parameters, qu_mol_graph):
      #  print("actionspace.py - class RemoveGroupActionSpace - method get_valid_actions_mask")

        # Initialization of the action space
        valid_action_space = np.full((parameters.max_heavy_atoms, parameters.max_heavy_atoms), False)

        # Extraction of the bridge matrix
        bridge_bond_matrix = qu_mol_graph.get_bridge_bonds_matrix()

        # Iterating over all bonds
        for i in range(qu_mol_graph.get_n_atoms()):
            for j in range(i + 1, qu_mol_graph.get_n_atoms()):

                # The group can be removed only if the bond is a bridge and none of the atoms has a formal charge
                # and at least one atom is mutable
                if bridge_bond_matrix[i][j] and qu_mol_graph.get_formal_charge(i) == 0 and \
                        qu_mol_graph.get_formal_charge(j) == 0 \
                        and (qu_mol_graph.get_atom_mutability(i) or qu_mol_graph.get_atom_mutability(j)):

                    # Extracting the indices of both connected components if the current bond was removed
                    connected_component_i = self._get_connected_component_after_removing_bond(qu_mol_graph, i, j)
                    connected_component_j = self._get_connected_component_after_removing_bond(qu_mol_graph, j, i)

                    # If the self.only_remove_smallest_group option is set to True, only the smallest component can be
                    # removed. Otherwise, both parts of the bond can be removed
                    if self.only_remove_smallest_group:
                        if len(connected_component_i) <= len(connected_component_j):
                            valid_action_space[i][j] = True

                        if len(connected_component_j) <= len(connected_component_i):
                            valid_action_space[j][i] = True
                    else:
                        valid_action_space[i][j] = True
                        valid_action_space[j][i] = True

        return valid_action_space.flatten()

    def get_action_space_size(self, parameters, qu_mol_graph):
        """
        The action space size is the maximum number of bonds (max_heavy_atoms*max_heavy_atoms)
        """
        return parameters.max_heavy_atoms * parameters.max_heavy_atoms

    def get_action_expl(self, id_action, parameters, qu_mol_graph):
        return ""

    def get_context(self, action_id, parameters, *args, **kwargs):
        raise Exception("Method not implemented yet")

    def execute_action(self, action_id, parameters, qu_mol_graph):
       # print("actionspace.py - class RemoveGroupActionSpace - method execute_action")
        #print("parameters :", parameters)
        #print("qu_mol_graph :", qu_mol_graph)
        #print("action_id :", action_id)
        
        smiles_before = qu_mol_graph.to_aromatic_smiles()

        i, j = self._action_id_to_atoms_idx(action_id, parameters)

        # Computing the indices of the atoms that are to be removed
        atoms_to_remove_indices = self._get_connected_component_after_removing_bond(qu_mol_graph, i, j)

        try:

            # Iterating over all atoms to be removed
            for at_idx in sorted(list(atoms_to_remove_indices), reverse=True):

                # Removing the current atom from the molecular graph
                qu_mol_graph.rm_atom(at_idx, False)

            # Updating the state of the MolGraph object
            qu_mol_graph.update_mol_representation()

        except Exception as e:
            print("Remove group caused error")
            print("Smiles before : " + str(smiles_before))
            print("Smiles after : " + str(qu_mol_graph.to_aromatic_smiles()))
            raise e

    def action_to_str(self, action_id, parameters, qu_mol_graph):
        return ""


class ChangeBondActionSpace(ActionSpace):
    """
    Changing a bond from any type to any type among no bond, single, double, triple.
    """

    def __init__(self, check_validity=True, keep_connected=False, prevent_removing_creating_bonds=False):
        """
        Changing the type of a bond
        :param check_validity: whether to check if the action is legal before application
        :param keep_connected: whether to make sure that actions cannot break the graph in multiple connected components
        :param prevent_removing_bonds: whether to prevent the change of bonds from type >= 1 to type 0 (=breaking
        existing bonds).
        """
        super().__init__(check_validity)
        self.keep_connected = keep_connected
        self.prevent_removing_creating_bonds = prevent_removing_creating_bonds

    def action_space_type_id(self):
        return "ChB"

    def _action_id_to_atoms_idx(self, action_id, parameters):
        """
        Converting the id of action to the indices of both atoms involved
        @return:
        """

        action_id_first_matrix_equivalence = action_id % (parameters.max_heavy_atoms *
                                                          (parameters.max_heavy_atoms - 1) // 2)
        #print("action_id :", action_id)
        #print ("CHB action_id_first_matrix_equivalence :", action_id_first_matrix_equivalence)
        #print ("indices matrice triangulaire :", np.triu_indices(parameters.max_heavy_atoms, k=1))
        #print ("indices matrice triangulaire [0] [i]:", np.triu_indices(parameters.max_heavy_atoms, k=1)[0][action_id_first_matrix_equivalence])
        #print ("indices matrice triangulaire [1] [i]:", np.triu_indices(parameters.max_heavy_atoms, k=1)[1][action_id_first_matrix_equivalence])
        
        return np.triu_indices(parameters.max_heavy_atoms, k=1)[0][action_id_first_matrix_equivalence], \
               np.triu_indices(parameters.max_heavy_atoms, k=1)[1][action_id_first_matrix_equivalence]

    def _action_id_to_bond_to_form(self, action_id, parameters):
        """
        Converting the id of the action to the type of the bond to be formed
        """

        return action_id // (parameters.max_heavy_atoms * (parameters.max_heavy_atoms - 1) // 2)

    def get_action_expl(self, action_id, parameters, qu_mol_graph):

        # Computing indices of both atoms involved in the action
        at1_idx, at2_idx = self._action_id_to_atoms_idx(action_id, parameters)

        # Extracting types of the couple of atoms
        first_at_type = qu_mol_graph.get_atom_type(at1_idx)
        second_at_type = qu_mol_graph.get_atom_type(at2_idx)

        # Extracting the numerical bond type between the atoms and the bond type that will be formed
        curr_bond_type_num = qu_mol_graph.get_bond_type_num(at1_idx, at2_idx)
        bond_to_form_type_num = self._action_id_to_bond_to_form(action_id, parameters)

        return first_at_type + ":" + second_at_type + ":" + str(curr_bond_type_num) + ":" + str(bond_to_form_type_num)

    def get_valid_actions_mask(self, parameters, qu_mol_graph):
       # print("actionspace.py - class ChangeBondActionSpace - method get_valid_actions_mask")

        # Add bond action space mask initialization as a (nb_bond_types, max_heavy_atoms, max_heavy_atoms), matrix initialized with "False" values
        add_bond_action_space_mask = np.full((4, parameters.max_heavy_atoms, parameters.max_heavy_atoms), False)

        # Extracting the vector of free electron for each atom
        free_electons_vect = qu_mol_graph.get_free_electrons_vector()

        # Extracting the vector of formal charges
        formal_charge_vect = qu_mol_graph.get_formal_charge_vector()

        if self.keep_connected:
            # as_multigraph is set to True because double+ bonds are not bridges in case of removal of a single electron
            # from the bond
            bridge_matrix = qu_mol_graph.get_bridge_bonds_matrix()

        # Computing the upper triangular matrix for each type of bonds to be formed
        for i in range(len(free_electons_vect)):
            for j in range(i + 1, len(free_electons_vect)):

                # Extracting current bond type
                curr_bond = qu_mol_graph.get_bond_type_num(i, j)

                # Iterating over all bonds to be formed
                for bond_to_form in range(4):

                    delta_bond = bond_to_form - curr_bond

                    formal_charge_ok = formal_charge_vect[i] == 0 and formal_charge_vect[j] == 0
                    mutability_ok = qu_mol_graph.get_atom_mutability(i) or qu_mol_graph.get_atom_mutability(j)

                    # Bond decrement
                    # If keep connected is set to True, only bond that are not bridges can be completely removed.
                    # Bond involving atoms with formal charges cannot be changed. Bonds can be changed only if at least
                    # one of the atoms is mutable
                    if delta_bond < 0:

                        # Checking if the prevent breaking bonds constraint is respected if set
                        prevent_breaking_bonds_constraint_respected = not self.prevent_removing_creating_bonds or bond_to_form > 0

                        if not self.keep_connected:
                            add_bond_action_space_mask[bond_to_form][i][j] = formal_charge_ok and mutability_ok and \
                                                                             prevent_breaking_bonds_constraint_respected
                        else:
                            add_bond_action_space_mask[bond_to_form][i][j] = (not bridge_matrix[i][
                                j] or bond_to_form > 0) \
                                                                             and formal_charge_ok and mutability_ok and \
                                                                             prevent_breaking_bonds_constraint_respected

                    # Bond increment
                    # Bond can be incremented of delta if each atom involved has at least delta free electrons
                    # Bonds involving atoms with formal charges cannot be changed
                    elif delta_bond > 0:

                        # Checking if the prevent creating bonds constraint is respected if set
                        prevent_breaking_bonds_constraint_respected = not self.prevent_removing_creating_bonds or curr_bond > 0

                        add_bond_action_space_mask[bond_to_form][i][j] = (min(free_electons_vect[i],
                                                                              free_electons_vect[j]) >= delta_bond) \
                                                                         and formal_charge_ok and mutability_ok and prevent_breaking_bonds_constraint_respected

        final_action_space = []

        for i in range(4):
            final_action_space.append(add_bond_action_space_mask[i][np.triu_indices(parameters.max_heavy_atoms, k=1)])

        # Returning the values of the upper triangular matrix
        return np.array(final_action_space).flatten()

    def get_action_space_size(self, parameters, qu_mol_graph):

        # The action space is of size sum from 1 to max_heavy_atoms -1 multiplied by the number of possible bond types
        return (parameters.max_heavy_atoms * (parameters.max_heavy_atoms - 1) // 2) * 4

    def get_context(self, action_id, parameters, *args, **kwargs):
        if not 'qu_mol_graph' in kwargs:
            raise Exception("qu_mol_graph parameter cannot be of type None for the ChangeBondActionSpace class")

        # Computing indices of both atoms involved in the action
        at1_idx, at2_idx = self._action_id_to_atoms_idx(action_id, parameters)

        # Extracting the numerical bond type between the atoms and the bond type that will be formed
        curr_bond_type_num = kwargs['qu_mol_graph'].get_bond_type_num(at1_idx, at2_idx)
        bond_to_form_type_num = self._action_id_to_bond_to_form(action_id, parameters)

        return [at1_idx, at2_idx, curr_bond_type_num, bond_to_form_type_num]

    def execute_action(self, action_id, parameters, qu_mol_graph):
        super(ChangeBondActionSpace, self).execute_action(action_id, parameters, qu_mol_graph)
       # print("actionspace.py - class ChangeBondActionSpace - method execute_action")
        #print("parameters :", parameters)
        #print("qu_mol_graph :", qu_mol_graph)
       # print("action_id :", action_id)
        
        at1_idx, at2_idx = self._action_id_to_atoms_idx(action_id, parameters)
       # print("at1_idx, at2_idx : ", at1_idx, " ,", at2_idx)
        bond_to_form = self._action_id_to_bond_to_form(action_id, parameters)
       # print('bond_to_form :', bond_to_form)
        try:
            # Performing bond addition
            qu_mol_graph.set_bond(at1_idx, at2_idx, bond_to_form)

        except Exception as e:
            print("Change bond caused error")
            raise e

        # Non terminal action
        return True

    def action_to_str(self, action_id, parameters, qu_mol_graph):

        at1_idx, at2_idx = self._action_id_to_atoms_idx(action_id, parameters)

        curr_bond = qu_mol_graph.get_bond_type_num(at1_idx, at2_idx)
        new_bond_type = self._action_id_to_bond_to_form(action_id, parameters)
        
        dscr = "Change bond between atoms of ids " + str(at1_idx) + " (" + qu_mol_graph.get_atom_type(
            at1_idx) + ")" + " and " + str(at2_idx) + " (" + qu_mol_graph.get_atom_type(at2_idx) + ") from " + str(curr_bond) + " to " + \
               str(new_bond_type)
               
      #  print("Action performed :", dscr)

        return dscr


class SubstituteAtomActionSpace(ActionSpace):
    """
    Defining the action space for atom substitution (defined atom type change)
    In order for a substitution to be valid, the atom on which the substitution applies must be defined and mutable. In
    addition, the substitution must be allowed in the parameters.accepted_substitutions dictionary.
    The substitution action space has a size of parameters.max_heavy_atoms * |parameters.accepted_atoms|
    """

    def action_space_type_id(self):
        return "Sub"

    def get_action_expl(self, action_id, parameters, qu_mol_graph):

        # Computing coordinates in the action space from action id
        i = action_id // len(parameters.accepted_atoms)
        j = action_id % len(parameters.accepted_atoms)

        curr_at_type = qu_mol_graph.get_atom_type(i)
        new_at_type = parameters.accepted_atoms[j]

        return curr_at_type + ":" + new_at_type

    def get_valid_actions_mask(self, parameters, qu_mol_graph):
      #  print("actionspace.py - class SubstituteAtomActionSpace - method get_valid_actions_mask")

        # Validity mask initialization
        substitute_valid_mask = np.full((parameters.max_heavy_atoms, len(parameters.accepted_atoms)), False)

        # Extracting the types of the atoms contained in the molecular graph
        mol_at_types = qu_mol_graph.get_atom_types()

        # Allowing specified substitutions
        for curr_atom_type, curr_allowed_substitutions in parameters.accepted_substitutions.items():

            # Extracting the list of index of atoms of type corresponding to the current type
            curr_at_type_idx = np.nonzero(mol_at_types == curr_atom_type)

            # Extracting the list of unallowed substitution for current atom type (numerical reference for atom types
            # is the index in the parameters.accepted_atoms list)
            allowed_subst_idx = []
            for i, at_type in enumerate(parameters.accepted_atoms):
                if at_type in curr_allowed_substitutions:
                    allowed_subst_idx.append(i)

            substitute_valid_mask[curr_at_type_idx, np.array(allowed_subst_idx).reshape(-1, 1)] = True

        # Extracting max valence of accepted atoms and repeating into a matrix
        max_valence_matrix = np.zeros((parameters.max_heavy_atoms, len(parameters.accepted_atoms)))
        max_valence_matrix[:qu_mol_graph.get_n_atoms()] = np.tile([qu_mol_graph.get_max_valence(at_type) for at_type in
                                                                   parameters.accepted_atoms],
                                                                  qu_mol_graph.get_n_atoms()).reshape(-1, len(
            parameters.accepted_atoms))

        # Extracting explicit valence of defined atom and repeating to match the number of accepted atoms
        expl_valence_matrix = np.zeros((parameters.max_heavy_atoms, len(parameters.accepted_atoms)))
        expl_valence_matrix[:qu_mol_graph.get_n_atoms()] = np.repeat(
            [qu_mol_graph.get_atom_degree(i, as_multigraph=True) for i in
             range(qu_mol_graph.get_n_atoms())],
            len(parameters.accepted_atoms)).reshape(-1, len(parameters.accepted_atoms))

        # Discarding valence incompatible substitutions
        substitute_valid_mask = np.logical_and(substitute_valid_mask,
                                               np.subtract(max_valence_matrix, expl_valence_matrix) >= 0)

        # Discarding substitution of undefined atoms
        substitute_valid_mask[qu_mol_graph.get_n_atoms():] = False

        # Discarding substitution of non mutable atoms
        for i in range(qu_mol_graph.get_n_atoms()):
            if not qu_mol_graph.get_atom_mutability(i):
                substitute_valid_mask[i] = False

        return substitute_valid_mask.reshape((-1,))

    def get_action_space_size(self, parameters, qu_mol_graph):
        return len(parameters.accepted_atoms) * parameters.max_heavy_atoms

    def get_context(self, action_id, parameters, *args, **kwargs):
        raise Exception("Method not implemented yet")

    def execute_action(self, action_id, parameters, qu_mol_graph):
        super(SubstituteAtomActionSpace, self).execute_action(action_id, parameters, qu_mol_graph)
      #  print("actionspace.py - class SubstituteAtomActionSpace - method execute_action")
        #print("parameters :", parameters)
        #print("qu_mol_graph :", qu_mol_graph)
        #print("action_id :", action_id)
        #print("len(parameters.accepted_atoms) :", len(parameters.accepted_atoms))
        
        at_id = action_id // len(parameters.accepted_atoms)
       # print("at_id :", at_id )
        
        new_type = parameters.accepted_atoms[action_id % len(parameters.accepted_atoms)]
       # print("action_id % len(parameters.accepted_atoms) : ", action_id % len(parameters.accepted_atoms))
       # print("new_type  :", new_type)

        try:
            qu_mol_graph.replace_atom(at_id, new_type)
        except Exception as e:
            print("Substitute atom type caused error")
            raise e

        # Non terminal action
        return True

    def action_to_str(self, action_id, parameters, qu_mol_graph):
        at_id = action_id // len(parameters.accepted_atoms)
        return "Substitute atom of id " + str(at_id) + " and type " + qu_mol_graph.get_atom_type(at_id) + " by " + \
               parameters.accepted_atoms[action_id % len(parameters.accepted_atoms)]
