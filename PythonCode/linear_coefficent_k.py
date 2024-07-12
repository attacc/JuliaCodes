import numpy as np
# Function-1: Derived definition of linear coefficient
def Linear_Coefficient_k(h_dim, s_dim, omega_space, eigval, dipole, eta):
    lin_coff = np.zeros((s_dim, len(omega_space)),dtype=np.cdouble)
    for a in range(s_dim):
        for w_idx, w in enumerate(omega_space):
            X1 = 0.0 + 0.0j
            for i in range(h_dim):
                E_i = eigval[i]
                for j in range(h_dim):
                  if j != i:
                    E_j = eigval[j]
                    X1 += abs(dipole[a, i, j])**2/ (E_i - E_j - w- 1j*eta)
            lin_coff[a, w_idx] = X1
    return lin_coff
