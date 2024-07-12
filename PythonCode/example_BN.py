#!/usr/bin/python
import sys
import math as M
import numpy as np
from hamiltonian import Hamiltonian,fix_eigenvec_phase
from lattice import set_Lattice
from gradH import gradH
from TB_tools import W2H_rotate,H2W_rotate
from linear_coefficent_k import Linear_Coefficient_k
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator


print("\n\n Example hBN tight-binding \n\n")

# Parameters
ha2ev=27.211396132              #Hartree to eV
E_g=3.625*2.0/ha2ev             #Energy Gap
t_0=2.30/ha2ev                  #NN hopping parameter

h_dim=2                         #Hamiltonian dimension
s_dim=2                         #Spatial dimension

a_cc=2.632                      #Lattice constant
Num_k=64                        #Number of k points

dk=1e-3
eta=0.1/ha2ev

# Create a 2x2 matrix with zeros
delta = np.full((3,s_dim),1.0)
a = np.full((s_dim,s_dim),1.0)
H = np.zeros((h_dim, h_dim), dtype=np.cdouble)

eigenvectors = np.zeros((h_dim, h_dim), dtype=np.cdouble)
eigenvalues = np.zeros((1, h_dim), dtype=np.cdouble)


max_omega=15.0/ha2ev
omega_space = np.linspace(0.0, max_omega, 400)
chi=np.zeros((s_dim,len(omega_space)),dtype=np.cdouble)
  
Dip_H=np.zeros((s_dim,h_dim,h_dim),dtype=np.complex)

#Lattice vectors
a[0,:]=(a_cc/2.0)*np.array([3.0,  M.sqrt(3.0)])
a[1,:]=(a_cc/2.0)*np.array([3.0, -M.sqrt(3.0)])

# Nearest Neighbour vectors:
delta[0,:]=(a_cc/2.0)*np.array([1.0,  M.sqrt(3.0)])
delta[1,:]=(a_cc/2.0)*np.array([1.0, -M.sqrt(3.0)])
delta[2,:]=(a_cc)*np.array([-1, 0])

lattice=set_Lattice(2,a)

for ik1 in range(Num_k):
    for ik2 in range(Num_k):
        vec_k=lattice.rvectors[0]*ik1/Num_k+lattice.rvectors[1]*ik2/Num_k

        print(" ik1 ",ik1," ik2 ",ik2," vec_k ",vec_k)
        
        print("  * * * Hamiltonian * * * ")
        H=Hamiltonian(h_dim,vec_k,a_cc,E_g,t_0,delta)
        print("H(1,1) ",H[0,0]*ha2ev)
        print("H(1,2) ",H[0,1]*ha2ev)
        eigenval, eigenvec = np.linalg.eigh(H)
        fix_eigenvec_phase(eigenvec, h_dim)
        
        print("  * * * Gradients of Hamiltonian * * * ")
        dH_k=gradH(h_dim,vec_k,a_cc,E_g,t_0,delta,dk)
        print("dH(1,2)/dx  = ",dH_k[0,0,1])
        print("dH(1,2)/dy  = ",dH_k[1,0,1])
        
        print("  * * * Dipoles * * * ")
        for a in range(s_dim):
            Dip_H[a,:,:]=W2H_rotate(dH_k[a,:,:],eigenvec)
        for n in range(h_dim):
          for m in range(n+1,h_dim):
            Dip_H[:,m, n] = 1j * Dip_H[:,m, n] / (eigenval[n] - eigenval[m])
            Dip_H[:,n, m] = np.conj(Dip_H[:,m, n])
        print("Dip_hx(1,2)  = ",np.real(Dip_H[0,0,1]),np.imag(Dip_H[0,0,1]))
        print("Dip_hy(1,2)  = ",np.real(Dip_H[1,0,1]),np.imag(Dip_H[1,0,1]))
        chi+=Linear_Coefficient_k(h_dim, s_dim, omega_space, eigenval, Dip_H, eta=eta) 

chi/=Num_k**2


# Plot results
fig, axs = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Electric susceptiblity first order plot', fontsize=20)
for s in range(s_dim):
    axs[s].set_xlim([0,max_omega*ha2ev])
    axs[s].plot(omega_space*ha2ev, np.real(chi[s, :]), label=f'Real Part')
    axs[s].plot(omega_space*ha2ev, np.imag(chi[s, :]), label=f'Imaginary Part')
    axs[s].set_xlabel('$\omega$')
    if s == 0:
      axs[s].set_ylabel('$\chi^{(1)}_x$')
    else:
      axs[s].set_ylabel('$\chi^{(1)}_y$')
    axs[s].legend(loc='upper right', fontsize='small')
    axs[s].grid(True)

plt.tight_layout()
plt.show()

