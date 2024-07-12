# Function-1: Derived definition of linear coefficient
def linear_coefficient_k(h_dim, omega_space, eigval, eigvec, vec_k, a_cc, E_g, t_0, delta, eta=0+1e-4j):
    s_dim=len(vec_k)
    dk_psi = gradU_H(h_dim, vec_k, a_cc, E_g, t_0, delta)[1]
    lin_coff = np.zeros((s_dim, len(omega_space)),dtype=complex)
    E_vec=[0.0,0.1]
    for a in range(s_dim):
        for w_idx, w in enumerate(omega_space):
            X1 = 0.0 + 0.0j
            for i in range(h_dim):
                E_i = eigval[i]
                u_i = eigvec[:, i]
                for j in range(h_dim):
                  if j != i:
                    E_j = eigval[j]
                    X1 += abs(np.vdot(u_i, dk_psi[a, :, j]))**2/ (E_i - E_j - w- eta)
            lin_coff[a, w_idx] = X1
    return lin_coff
