try:
    import numpy as np
    from numba import jit
    import numba.core.types
    import numba.typed
    import numba
    from hmm_profile.models import HMM
    import time
    import graph_tool.topology
except ImportError:
    print('[Error] Seems you do not have the required python packages. Please check it.')

# stndard library
from math import log
from typing import List, Dict, Tuple
import sys

from snakehead import DAG, Idx, timeit

class PHMM:
    """The class encapsulating HMM_profile HMM object.

    The overall structure of this class is adapted from https://github.com/janmax/Profile-HMM.

    Attributes:
        _phmm: A HMM_profile HMM object.
        _alphabet: A list of alphabets used in HMM. In this module, protein code.
        _alphabet_to_index: A list of index coverted from each protein code.
        _len: Length of the residues in HMM.
        _transmissions: A numpy array containing the transmission probabilities from states to connected states.
        _emissions_from_M: A numpy array containing the match state emission probabilities at states.
        _emissions_from_I: A numpy array containing the insertion state emission probabilities at states.
    """

    def __init__(self, phmm_ : HMM):
        """Inits PHMM with a given HMM object."""
        # Getting the length
        self._phmm : HMM = phmm_
        self._alphabet : List[str] = [x for x in self._phmm.metadata.alphabet]
        self._len : int = self._phmm.metadata.length

        # Transferting the probabilities
        self._transmissions : np.ndarray = self.transfer_transmissions()
        self._emissions_from_M : np.ndarray
        self._emissions_from_I : np.ndarray
        self._emissions_from_M, self._emissions_from_I = self.transfer_emissons()

    def __len__(self) -> int:
        """Return the number of positions in the HMM."""
        return self._len

    @timeit
    def viterbi(self, dag_ : DAG, hmm_from_ : int, ali_idxs_ : Tuple[int, int]):
        """Return the path corrected with viterbi algorithm.
           Generate data objects to store predecessors, ancestors and base of each node where every node id is converted into index ordered with topological sort for numpy operation.    
        """

        predecessors : np.ndarray = np.zeros((len(dag_), len(dag_) - 1), dtype=np.int32)
        bases : np.array = np.empty(len(dag_), dtype=np.unicode_ )

        for i in range(len(dag_)):

            node_id = dag_.index_to_node_id(i)
            pred = dag_.predecessors(node_id)
            for j in range(len(dag_.predecessors(node_id))):
                predecessors[i][j] = dag_.node_id_to_index(pred[j])
            bases[i] = dag_.base(node_id)

        V_M_tr, V_I_tr, V_D_tr, max_tr_idx = self._viterbi(
            predecessors,
            bases,
            self._emissions_from_M,
            self._emissions_from_I,
            self._transmissions,
            len(dag_),
            len(self),
            hmm_from_,
            ali_idxs_[0],
            ali_idxs_[1]
        )

        tr = [V_M_tr, V_I_tr, V_D_tr]
        corrected_path = [ dag_.index_to_node_id(x) for x in self.traceback(tr, ali_idxs_[0], max_tr_idx) ]

        return corrected_path

    @timeit
    def traceback(self, tr_, ali_from_, tr_start_idx_):
        """Trace back the traceback matrix so that we could identify the path with best score."""

        traceback_index_list = []

        t, i,j = tr_start_idx_

        while i > ali_from_:
            traceback_index_list.append(i)

            t, i,j = (tr_[t][i][j][0], tr_[t][i][j][1], tr_[t][i][j][2])

        # begin node
        traceback_index_list.append(i)
        
        traceback_index_list.reverse()
        return traceback_index_list

    def transfer_emissons(self) -> np.ndarray:
        """Transfer the emission probabilites into numpy arrays from HMM_profile HMM object."""
        emissions_from_M : Dict[str, np.ndarray] = {char: np.zeros(len(self)+1) for char in self._alphabet}
        emissions_from_I : Dict[str, np.ndarray] = {char: np.zeros(len(self)+1) for char in self._alphabet}

        for i, alphabet in enumerate(self._alphabet):
            emissions_from_M[alphabet][0] = self._phmm.start_step.p_emission_char[i]
            emissions_from_I[alphabet][0] = self._phmm.start_step.p_insertion_char[i]

        for i, alphabet in enumerate(self._alphabet):
            for j in range(1, len(self)+1):
                emissions_from_M[alphabet][j] = self._phmm.steps[j-1].p_emission_char[i]
                emissions_from_I[alphabet][j] = self._phmm.steps[j-1].p_insertion_char[i]

        # return 2D arrays for performance
        return \
            np.vstack([emissions_from_M[c] for c in self._alphabet]), \
            np.vstack([emissions_from_I[c] for c in self._alphabet])
    
    def transfer_transmissions(self) -> np.ndarray:
        """Transfer the transmission probabilites into numpy arrays from HMM_profile HMM object."""
        # these are all the transmissions we want to observe
        transmission_list = [
            'm->m', 'm->i', 'm->d', 'i->m', 'i->i', 'd->m', 'd->d'
            ]
        transmissions : Dict[str, np.ndarray]= {t: np.zeros(len(self)+1) for t in transmission_list}
        
        transmissions['m->m'][0] = self._phmm.start_step.p_emission_to_emission
        transmissions['m->i'][0] = self._phmm.start_step.p_emission_to_insertion
        transmissions['m->d'][0] = self._phmm.start_step.p_emission_to_deletion
        transmissions['i->m'][0] = self._phmm.start_step.p_insertion_to_emission
        transmissions['i->i'][0] = self._phmm.start_step.p_insertion_to_insertion

        for i in range(1, len(self)+1):
            transmissions['m->m'][i] = self._phmm.steps[i-1].p_emission_to_emission
            transmissions['m->i'][i] = self._phmm.steps[i-1].p_emission_to_insertion
            transmissions['m->d'][i] = self._phmm.steps[i-1].p_emission_to_deletion
            transmissions['i->m'][i] = self._phmm.steps[i-1].p_insertion_to_emission
            transmissions['i->i'][i] = self._phmm.steps[i-1].p_insertion_to_insertion
            transmissions['d->m'][i] = self._phmm.steps[i-1].p_deletion_to_emission
            transmissions['d->d'][i] = self._phmm.steps[i-1].p_deletion_to_deletion

        # return everything as a 2D array for performance
        return np.vstack([transmissions[t] for t in transmission_list])

    @staticmethod
    @jit(nopython=True)
    def _viterbi(predecessors_ : np.ndarray, bases_ :np.array, e_M_ : np.ndarray, e_I_ : np.ndarray, a_ : np.ndarray, N_ : int, L_ : int, hmm_from_ : int, ali_from_ : int, ali_to_ : int):
        """Inner function for Viterbi algorithm.
        """
        V_M : np.ndarray = np.zeros((N_, L_))
        V_I : np.ndarray = np.zeros((N_, L_))
        V_D : np.ndarray = np.zeros((N_, L_))

        V_M_tr : np.ndarray = np.zeros((N_, L_, 3), dtype=np.int32) # first: type of alignment 0 - M, 1 - I, 2 - D, 3 - N, 4 - C
        V_I_tr : np.ndarray = np.zeros((N_, L_, 3), dtype=np.int32) # second: DAG node index
        V_D_tr : np.ndarray = np.zeros((N_, L_, 3), dtype=np.int32) # third: hmm residue index

        curr_base = bases_[ali_from_]
        if curr_base == "A":
            x = 0
        elif curr_base == "C":
            x = 1
        elif curr_base == "G":
            x = 2
        else: # curr_base == "T"
            x = 3

        V_M[ali_from_][hmm_from_] = 10000000000

        for i in range(ali_from_+1, ali_to_+1): # Node index in topological order
            curr_base = bases_[i]

            if i == N_-1 or curr_base =='^' or curr_base == '$':
                continue

            else:
                if curr_base == "A":
                    x = 0
                elif curr_base == "C":
                    x = 1
                elif curr_base == "G":
                    x = 2
                else: # curr_base == "T"
                    x = 3

                for p in predecessors_[i]: # x_i^(1)
                    if p == 0:
                        continue
                    assert p < i

                    for j in range(hmm_from_+1, L_): # HMM residue index

                        if j != 0: # skip first residue
                            m_to_m = log(e_M_[x][j]) - log(e_M_[x][0]) + V_M[p][j-1] + log(a_[0][j]) # M->M
                            if m_to_m > V_M[i][j]: 
                                V_M[i][j] = m_to_m
                                V_M_tr[i][j][0] = 0
                                V_M_tr[i][j][1] = p
                                V_M_tr[i][j][2] = j-1

                            i_to_m = log(e_M_[x][j]) - log(e_M_[x][0]) + V_I[p][j-1] + log(a_[3][j]) # I->M
                            if i_to_m > V_M[i][j]: 
                                V_M[i][j] = i_to_m
                                V_M_tr[i][j][0] = 1
                                V_M_tr[i][j][1] = p
                                V_M_tr[i][j][2] = j-1

                            d_to_m = log(e_M_[x][j]) - log(e_M_[x][0]) + V_D[p][j-1] + log(a_[5][j]) # D->M
                            if d_to_m > V_M[i][j]: 
                                V_M[i][j] = d_to_m
                                V_M_tr[i][j][0] = 2
                                V_M_tr[i][j][1] = p
                                V_M_tr[i][j][2] = j-1

                        m_to_i = log(e_I_[x][j]) - log(e_I_[x][0]) + V_M[p][j] + log(a_[1][j]) # M->I
                        if m_to_i > V_I[i][j]:
                            V_I[i][j] = m_to_i
                            V_I_tr[i][j][0] = 0
                            V_I_tr[i][j][1] = p
                            V_I_tr[i][j][2] = j

                        i_to_i = log(e_I_[x][j]) - log(e_I_[x][0]) + V_I[p][j] + log(a_[4][j]) # I->I
                        if i_to_i > V_I[i][j]:
                            V_I[i][j] = i_to_i
                            V_I_tr[i][j][0] = 1
                            V_I_tr[i][j][1] = p
                            V_I_tr[i][j][2] = j

                        if j != 0 and j != L_-1: # skip first and last residues
                            m_to_d = V_M[i][j-1] + log(a_[2][j]) # M->D
                            if m_to_d > V_D[i][j]:
                                V_D[i][j] = m_to_d
                                V_D_tr[i][j][0] = 0
                                V_D_tr[i][j][1] = i
                                V_D_tr[i][j][2] = j-1

                            d_to_d = V_D[i][j-1] + log(a_[6][j]) # D->D
                            if d_to_d > V_D[i][j]:
                                V_D[i][j] = d_to_d
                                V_D_tr[i][j][0] = 2
                                V_D_tr[i][j][1] = i
                                V_D_tr[i][j][2] = j-1

        max_idx = np.argmax(V_M[ali_to_][:])

        return V_M_tr, V_I_tr, V_D_tr, (0,ali_to_,max_idx)