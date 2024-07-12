#!/usr/bin/python
import sys
import math as M
import numpy as np
from hamiltonian import Hamiltonian
from lattice import set_Lattice

print("\n\n Example hBN tight-binding \n\n")

# Parameters
ha2ev=27.211396132              #Hartree to eV
E_g=3.625*2.0/ha2ev             #Energy Gap
t_0=2.30/ha2ev                  #NN hopping parameter

h_dim=2                         #Hamiltonian dimension
s_dim=2                         #Spatial dimension

a_cc=2.632                      #Lattice constant
Num_k=2                        #Number of k points


# Create a 2x2 matrix with zeros
delta = np.full((3,s_dim),1.0)
a = np.full((s_dim,s_dim),1.0)
H = np.zeros((h_dim, h_dim), dtype=np.cdouble)

eigenvectors = np.zeros((h_dim, h_dim), dtype=np.cdouble)
eigenvalues = np.zeros((1, h_dim), dtype=np.cdouble)

max_omega=2.0/ha2ev
omega_space = np.linspace(0.0, max_omega, 400)

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
        H=Hamiltonian(h_dim,vec_k,a_cc,E_g,t_0,delta)
        print("H(1,1) ",H[0,0]*ha2ev)
        print("H(1,2) ",H[0,1]*ha2ev)
        eigenvalues, eigenvectors = np.linalg.eigh(H)

