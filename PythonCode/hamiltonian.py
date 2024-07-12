import numpy as np


def e_k(t_0,vec_k,delta):
  # Initialize the result
  ek = 0+0j
  # Calculate the dot product of A and each vector in B, raise it to the power of 'e', and sum the results
  for i in range(delta.shape[0]):
    dot_product = np.dot(vec_k, delta[i,:])
    ek = ek+(np.exp(1j*dot_product))
  return -t_0*ek

def Hamiltonian(h_dim,vec_k, a_cc,E_g,t_0,delta):
  H = np.zeros((h_dim, h_dim), dtype=complex)
  ek=e_k(t_0,vec_k,delta)
  for i in range(0,h_dim):
    for j in range(0,h_dim):
      if(i==j) :
        H[i,j]=(-1)**i*E_g/2.0
      elif(i<j) :
        H[i,j]=ek
      else:
        H[i,j]=np.conjugate(ek)
  #Various interacting terms can be added
  if not np.allclose(H, H.conj().T):
    print("!!! The Hamiltonian is not Hermitian !!!!")
    return np.zeros((h_dim, h_dim), dtype=complex)
  else:
    return H


#Fixing eigenvector phase
def fix_eigenvec_phase(eigenvec,h_dim):
  phase_m = np.zeros(h_dim, dtype=complex)
  # Rotation phase matrix
  for i in range(h_dim):
    phase_m[i] = np.exp(-1j * np.angle(eigenvec[i, i]))
    # Apply phase corrections
    eigenvec[:, i] = eigenvec[:, i]*phase_m[i]
  return eigenvec

