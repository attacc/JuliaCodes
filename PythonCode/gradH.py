import numpy as np
from hamiltonian import Hamiltonian,fix_eigenvec_phase
#Hamiltonan gradient
def gradH(h_dim, k, a_cc, E_g, t_0, delta, dk=1e-3):
    s_dim=len(k)
    dH_dk = np.zeros((s_dim, h_dim, h_dim), dtype=complex)
    for i in range(s_dim):
        # Perturb along the i-th dimension
        k_perturb_pos  = np.copy(k) 
        k_perturb_neg = np.copy(k)
        k_perturb_pos[i]+=dk
        k_perturb_neg[i]-=dk
        # Compute eigenstates for perturbed and perturbed negative k
        vx_p = Hamiltonian(h_dim, k_perturb_pos, a_cc, E_g, t_0, delta)
        vx_n = Hamiltonian(h_dim, k_perturb_neg, a_cc, E_g, t_0, delta)
        # Numerical differentiation to find the gradient of eigenstates
        dH_dk[i] = (vx_p - vx_n) / (2 * dk)
    return dH_dk
