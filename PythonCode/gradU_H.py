#Wavefunction Gradient with Hamiltonian gauge
def gradU_H(h_dim, vec_k, a_cc, E_g, t_0, delta, dk=1e-3,eta=0.0+1e-5j):
    s_dim=len(vec_k)
    dH_k=gradH(h_dim,vec_k,a_cc,E_g,t_0,delta,dk)
    eigval, eigvec = np.linalg.eigh(Hamiltonian(h_dim,vec_k,a_cc,E_g,t_0,delta))
    off_diag = np.identity(h_dim, dtype=complex)
    dU_H=np.zeros((s_dim,h_dim,h_dim),dtype=complex)
    Dip_H=np.zeros((s_dim,h_dim,h_dim),dtype=complex)
    for a in range(s_dim):
      for n in range(h_dim):
        u_n=eigvec[:,n]
        E_n=eigval[n]
        s=np.zeros((1,h_dim))
        for m in range(h_dim):
          if m != n:
            u_m=eigvec[:,m]
            E_m=eigval[m]
            s=s+sandwich_product(u_m,WH_rotate(dH_k[a,:,:],eigvec),u_n)*u_m/(E_n-E_m)
        dU_H[a,n,:]=s
        Dip_H[a,:,:]=WH_rotate(dH_k[a,:,:],eigvec)
      for i in range(h_dim):
        for j in range(h_dim):
          Dip_H[a,i, j] = Dip_H[a,i, j]*off_diag[i,j]
    for n in range(h_dim):
      for m in range(n+1,h_dim):
        Dip_H[:,m, n] = 1j * Dip_H[:,m, n] / (eigval[m] - eigval[n])
        Dip_H[:,m, n] = np.conj(Dip_H[:,m, n])
    return Dip_H, dU_H


