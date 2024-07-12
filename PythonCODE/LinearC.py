import numpy as np
from fractions import Fraction as frac
import csv
import math as M
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import sys
import time
from time import sleep

def loading(start,end):
    sys.stdout.write('\r')
    sys.stdout.write("[%-10s] %d%%" % ('='*int(start*100/end), 1*int(start*100/end)))
    sys.stdout.flush()
    sleep(0) #sleep(0.02)

def dirac_delta(x):
  if x == 0:
    return 1
  else:
    return 0

def WH_rotate(M, eigenvec):
    return np.conjugate(eigenvec).T @ M @ eigenvec

def HW_rotate(M, eigenvec):
    return eigenvec @ M @ np.conjugate(eigenvec).T

def sandwich_product(p,A,q):
  # Ensure p, A, and q are numpy arrays
  p = np.array(p)
  A = np.array(A)
  q = np.array(q)
  result = np.dot(np.conjugate(p).T, np.dot(A, q))
  return result

def e_k(t_0,vec_k,delta):
  # Initialize the result
  ek = 0+0j
  # Calculate the dot product of A and each vector in B, raise it to the power of 'e', and sum the results
  for i in range(delta.shape[1]):
    dot_product = np.dot(vec_k, delta[:, i])
    ek = ek+(np.exp(1j*dot_product))
  return -t_0*ek

#Fermi-Dirac distribution
def fermi_dis(E_n,mu=0.5,b=38.94):
  return 1.0/(1.0+np.exp(b*(E_n-mu)))

def Hamiltonian(h_dim,vec_k,a_cc,E_g,t_0,delta):
  H = np.zeros((h_dim, h_dim), dtype=complex)
  #ek=np.exp(-1j*vec_k[0]*a_cc)*(1.0+2.0*np.exp(1j*vec_k[0]*3.0*a_cc/2.0)*M.cos(M.sqrt(3.0)*vec_k[1]*a_cc/2.0))
  ek=e_k(t_0,vec_k,delta)
  #ek=np.exp(1j*np.dot(vec_k,delta[:,0]))+np.exp(1j*np.dot(vec_k,delta[:,1]))+np.exp(1j*np.dot(vec_k,delta[:,2]))
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

#Wavefunction Gradient
def gradU(h_dim, k, a_cc, E_g, t_0, delta, dk=1e-3):
    s_dim=len(k)
    dpsi_dk = np.zeros((s_dim, h_dim, h_dim), dtype=complex)
    for i in range(s_dim):
        # Perturb along the i-th dimension
        k_perturb = k + dk * np.eye(s_dim)[:, i]
        k_perturb_neg = k - dk * np.eye(s_dim)[:, i]
        # Compute eigenstates for perturbed and perturbed negative k
        vx_p = np.linalg.eigh(Hamiltonian(h_dim, k_perturb, a_cc, E_g, t_0, delta))[1]
        vx_n = np.linalg.eigh(Hamiltonian(h_dim, k_perturb_neg, a_cc, E_g, t_0, delta))[1]
        fix_eigenvec_phase(vx_p, h_dim)
        fix_eigenvec_phase(vx_n, h_dim)
        # Numerical differentiation to find the gradient of eigenstates
        dpsi_dk[i,:,:] = (vx_p - vx_n) / (2 * dk)
    return dpsi_dk

#Hamiltonan gradient
def gradH(h_dim, k, a_cc, E_g, t_0, delta, dk=1e-3):
    s_dim=len(k)
    dH_dk = np.zeros((s_dim, h_dim, h_dim), dtype=complex)
    for i in range(s_dim):
        # Perturb along the i-th dimension
        k_perturb = k + dk * np.eye(s_dim)[:, i]
        k_perturb_neg = k - dk * np.eye(s_dim)[:, i]
        # Compute eigenstates for perturbed and perturbed negative k
        vx_p = Hamiltonian(h_dim, k_perturb, a_cc, E_g, t_0, delta)
        vx_n = Hamiltonian(h_dim, k_perturb_neg, a_cc, E_g, t_0, delta)
        # Numerical differentiation to find the gradient of eigenstates
        dH_dk[i] = (vx_p - vx_n) / (2 * dk)
    return dH_dk

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
            s=s+sandwich_product(u_m,WH_rotate(dH_k[a,:,:],eigvec),u_n)*u_m/(E_n-E_m+eta)
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


# Function-1: Derived definition of linear coefficient
def linear_coefficient(h_dim, omega_space, eigval, eigvec, vec_k, a_cc, E_g, t_0, delta, eta=0+1e-4j):
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
'''

# Function-2: Claudio's definition of linear coefficient
def linear_coefficient(h_dim, freqs, eigval, eigvec, vec_k, a_cc, E_g, t_0, delta, eta=0+1e-4j, nv=1):
    xhi = np.zeros(len(freqs), dtype=complex)
    Res = np.zeros((h_dim, h_dim), dtype=complex)
    Ef_vec=[0.0,0.1]
    Dip_H=gradU_H(h_dim, vec_k, a_cc, E_g, t_0, delta)[0]
    for iv in range(nv):
      for ic in range(nv, h_dim):
        Res[iv, ic] = np.sum(Dip_H[iv, ic, :] * Ef_vec[:])
        e_v = eigval[iv]
        e_c = eigval[ic]
        for ifreq, w in enumerate(freqs):
          xhi[ifreq] = abs(Res[iv, ic])**2 / (e_c - e_v - w - eta )
    xhi = xhi
    return xhi

# Function-3: Analytical definition of linear coefficient from notes
def linear_coefficient(h_dim, freqs, eigval, eigvec, vec_k, a_cc, E_g, t_0, delta, eta=0+1e-4j, nv=1):
    xhi = np.zeros((s_dim, len(freqs)), dtype=complex)
    dH_k=gradH(h_dim,vec_k,a_cc,E_g,t_0,delta)
    for a in range(s_dim):
      for i in range(h_dim):
        ei=eigval[i]
        u_i=eigvec[:,i]
        for j in range(h_dim):
          if j != i:
            ej=eigval[j]
            u_j=eigvec[:,j]
            numer=abs(sandwich_product(u_i,dH_k[a,:,:],u_j))**2*(fermi_dis(ei)-fermi_dis(ej))
            denomer=(ej-ei)**2
            for ifreq, w in enumerate(freqs):
              xhi[a,ifreq]=(numer/denomer)*(1.0/(w-ej-ei+eta))
    return xhi
'''
# Parameters
ha2ev=27.211396132              #Hartree to eV
E_g=3.625*2.0/ha2ev             #Energy Gap
t_0=2.30/ha2ev                  #NN hopping parameter
h_dim=2                         #Hamiltonian dimension
s_dim=2                         #Spatial dimension
a_cc=2.632                      #Lattice constant
Num_k=3 #401                       #Number of k points
k_1=-M.pi/a_cc                  #Starting k-value
k_2=-1*k_1                      #Ending k-value

# Create a 2x2 matrix with zeros
delta = np.full((s_dim,3),1.0)
a = np.full((s_dim,s_dim),1.0)
vec_k = np.full((s_dim),k_1)
H = np.zeros((h_dim, h_dim), dtype=complex)
eigenvectors = np.zeros((h_dim, h_dim), dtype=complex)
eigenvalues = np.zeros((1, h_dim), dtype=complex)
k_points_1D=np.linspace(k_1, k_2, Num_k)
max_omega=1.0
omega_space = np.linspace(0.0, max_omega, 400)
chi_r=np.zeros((s_dim,len(omega_space)),dtype=float)
chi_i=np.zeros((s_dim,len(omega_space)),dtype=float)
chi=np.zeros((Num_k,Num_k,s_dim,len(omega_space)),dtype=complex)

#Lattice vectors
a[:,0]=(a_cc/2.0)*np.array([3.0,  M.sqrt(3.0)])
a[:,1]=(a_cc/2.0)*np.array([3.0, -M.sqrt(3.0)])

# Nearest Neighbour vectors:
delta[:,0]=(a_cc/2.0)*np.array([1.0,  M.sqrt(3.0)])
delta[:,1]=(a_cc/2.0)*np.array([1.0, -M.sqrt(3.0)])
delta[:,2]=(a_cc)*np.array([-1, 0])



#=========================================================Processing===========================================================================
iter=0
for kp_x,vec_k[0] in enumerate(k_points_1D):
  loading(kp_x+1,Num_k)
  for kp_y,vec_k[1] in enumerate(k_points_1D):
    H=Hamiltonian(h_dim,vec_k,a_cc,E_g,t_0,delta)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    chi[kp_x,kp_y,:,:] = linear_coefficient(h_dim, omega_space, eigenvalues, eigenvectors, vec_k, a_cc, E_g, t_0, delta, eta=0+(0.15/ha2ev)*1j)
    for s in range(s_dim):
      chi_r[s,:]=chi_r[s,:]+chi[kp_x,kp_y,s,:].real/Num_k
      chi_i[s,:]=chi_i[s,:]+chi[kp_x,kp_y,s,:].imag/Num_k



#======================================================Data======================================================================================
print('\n \t')
for i,w in enumerate(omega_space):
  print("{3}) {0}, {1}, {2}".format(w, chi_r[0,i], chi_i[0,i], i+1))

print('\n \t')
for i,w in enumerate(omega_space):
  print("{3}) {0}, {1}, {2}".format(w, chi_r[1,i], chi_i[1,i], i+1))


#=========================================================Linear coefficient plot===============================================================
# Plot results
fig, axs = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Electric susceptiblity first order plot', fontsize=20)
chi_r=chi_r/float(Num_k)
chi_i=chi_i/float(Num_k)

for s in range(s_dim):
    axs[s].plot(omega_space, chi_r[s, :], label=f'Real Part')
    axs[s].plot(omega_space, chi_i[s, :], label=f'Imaginary Part')
    axs[s].set_xlabel('$\omega$')
    axs[s].set_xlim(0,max_omega)
    if s == 0:
      axs[s].set_ylabel('$\chi^{(1)}_x$')
    else:
      axs[s].set_ylabel('$\chi^{(1)}_y$')
    axs[s].legend(loc='upper right', fontsize='small')
    axs[s].grid(True)

plt.tight_layout()
plt.show()
