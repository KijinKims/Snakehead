import numpy as np
from numba import jit
import numba.core.types
import numba.typed
import numba
from numba.pycc import CC
from math import log

cc = CC('cppviterbi')

@cc.export('_viterbi', (numba.int32[:,:], numba.core.types.unicode_type[:], numba.float64[:,:], numba.float64[:,:], numba.float64[:,:], numba.int32, numba.int32))
def _viterbi(predecessors_ : np.ndarray, bases_ :np.array, e_M_ : np.ndarray, e_I_ : np.ndarray, a_ : np.ndarray, N_ : int, L_ : int):
    """Inner function for Viterbi algorithm.
    """
    V_M : np.ndarray = np.zeros((N_, L_))
    V_I : np.ndarray = np.zeros((N_, L_))
    V_D : np.ndarray = np.zeros((N_, L_))
    V_N : np.ndarray = np.zeros(N_)
    V_C : np.ndarray = np.zeros(N_)

    V_M_tr : np.ndarray = np.zeros((N_, L_, 3), dtype=np.int32) # first: type of alignment 0 - M, 1 - I, 2 - D, 3 - N, 4 - C
    V_I_tr : np.ndarray = np.zeros((N_, L_, 3), dtype=np.int32) # second: DAG node index
    V_D_tr : np.ndarray = np.zeros((N_, L_, 3), dtype=np.int32) # third: hmm residue index
    V_N_tr : np.ndarray = np.zeros((N_, 3), dtype=np.int32)
    V_C_tr : np.ndarray = np.zeros((N_, 3), dtype=np.int32)

    V_N[0] = 0

    for i in range(1, N_): # Node index in topological order
        curr_base = bases_[i]
        for p in predecessors_[i]: # x_i^(1)
            assert p < i

            n_to_n = V_N[p]    # N->N
            if n_to_n > V_N[i]:
                V_N[i] = n_to_n
                V_N_tr[i][0] = 3
                V_N_tr[i][1] = p
                V_N_tr[i][2] = 0

            max_idx = np.argmax(V_M[p][:])
            m_to_c = V_M[p][max_idx]   # M->C
            if m_to_c > V_C[i]:
                V_C[i] = m_to_c
                V_C_tr[i][0] = 0
                V_C_tr[i][1] = p
                V_C_tr[i][2] = max_idx

            c_to_c = V_C[p]   # C->C
            if c_to_c > V_C[i]:
                V_C[i] = c_to_c
                V_C_tr[i][0] = 4
                V_C_tr[i][1] = p
                V_C_tr[i][2] = 0

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
                assert p < i
                if p == 0:
                    break

                for j in range(L_): # HMM residue index

                    if j != 0: # skip first residue
                        m_to_m = log(e_M_[x][j+1]) - log(e_M_[x][0]) + V_M[p][j-1] + log(a_[0][j]) # M->M
                        if m_to_m > V_M[i][j]: 
                            V_M[i][j] = m_to_m
                            V_M_tr[i][j][0] = 0
                            V_M_tr[i][j][1] = p
                            V_M_tr[i][j][2] = j-1
                        
                        i_to_m = log(e_M_[x][j+1]) - log(e_M_[x][0]) + V_I[p][j-1] + log(a_[3][j]) # I->M
                        if i_to_m > V_M[i][j]: 
                            V_M[i][j] = i_to_m
                            V_M_tr[i][j][0] = 1
                            V_M_tr[i][j][1] = p
                            V_M_tr[i][j][2] = j-1
                        
                        d_to_m = log(e_M_[x][j+1]) - log(e_M_[x][0]) + V_D[p][j-1] + log(a_[5][j]) # D->M
                        if d_to_m > V_M[i][j]: 
                            V_M[i][j] = d_to_m
                            V_M_tr[i][j][0] = 2
                            V_M_tr[i][j][1] = p
                            V_M_tr[i][j][2] = j-1

                    n_to_m = log(e_M_[x][j+1]) - log(e_M_[x][0]) + V_N[p] # N->M
                    if n_to_m > V_M[i][j]: 
                        V_M[i][j] = n_to_m
                        V_M_tr[i][j][0] = 3
                        V_M_tr[i][j][1] = p
                        V_M_tr[i][j][2] = 0

                    m_to_i = log(e_I_[x][j+1]) - log(e_I_[x][0]) + V_M[p][j] + log(a_[1][j+1]) # M->I
                    if m_to_i > V_I[i][j]:
                        V_I[i][j] = m_to_i
                        V_I_tr[i][j][0] = 0
                        V_I_tr[i][j][1] = p
                        V_I_tr[i][j][2] = j
                    
                    i_to_i = log(e_I_[x][j+1]) - log(e_I_[x][0]) + V_I[p][j] + log(a_[4][j+1]) # I->I
                    if i_to_i > V_I[i][j]:
                        V_I[i][j] = i_to_i
                        V_I_tr[i][j][0] = 1
                        V_I_tr[i][j][1] = p
                        V_I_tr[i][j][2] = j

                    if j != 0 and j != L_-1: # skip first and last residues
                        m_to_d = V_M[i][j-1] + log(a_[2][j+1]) # M->D
                        if m_to_d > V_D[i][j]:
                            V_D[i][j] = m_to_d
                            V_D_tr[i][j][0] = 0
                            V_D_tr[i][j][1] = i
                            V_D_tr[i][j][2] = j-1
                        
                        d_to_d = V_D[i][j-1] + log(a_[6][j+1]) # D->D
                        if d_to_d > V_D[i][j]:
                            V_D[i][j] = d_to_d
                            V_D_tr[i][j][0] = 2
                            V_D_tr[i][j][1] = i
                            V_D_tr[i][j][2] = j-1

    return V_M_tr, V_I_tr, V_D_tr, V_N_tr, V_C_tr, (4,N_-1,0)

if __name__ == "__main__":
    cc.compile()