#
# Module for Tight-binding code for monolayer hexagonal boron nitride
# Claudio Attaccalite (2023)
#
#=
 Lattice vectors
 
 a_1=a/2 (3,  sqrt(3))
 a_2=a/2 (3, -sqrt(3))
 
 First neighboars
 
 delta_1 =  a/2 ( 1 ,  sqrt(3))
 delta_2 =  a/2 ( 1 , -sqrt(3))
 delta_3 =  a/2 (-2 , 0 )
 
 Volume = a_1 * a_2 = 3/2 a^2
  
 Atoms position
 atom_1=(0,0)
 atom_2=1/2(1, sqrt(3.0))

 Reciprocal space
 
 b_1 = 2\pi (a_1 ^ a_2) /V = 2\pi/3a (1,  sqrt(3)) 
 b_2 = 2\pi (a_2 ^ a_1) /V = 2\pi/3a (1, -sqrt(3))

 Special K-points

 K =  2\pi/3a (1,  1/sqrt(3))  
 K'= -2\pi/3a (1, -1/sqrt(3))
 M = b_1/2
 M'= b_2/2

=# 


module hBN2D

using LinearAlgebra
include("TB_tools.jl")
using .TB_tools
#
# In this program the lattice constant is equal to 1

export Hamiltonian,Berry_Connection,Dipole_Matrices,Calculate_UdU
  #
  global ndim=2
  #
  function Hamiltonian(k)::Matrix{Complex{Float64}}
        #
        # Default paramters from PRB 100, 195201 (2019)
        t_0 = 2.92 
        E_gap=5.62
        #
	H=zeros(Complex{Float64},2,2)
        #
        # Diagonal part 0,E_gap
        #
	H[1,1]= E_gap/2.0
	H[2,2]=-E_gap/2.0
        #
        # Off diagonal part
        # f(k)=e^{-i * k_x) * (1+2*e^{ i * k_x *3/2} ) * cos(sqrt3)/2 *k_y)
        #
	f_k=exp(-1im*k[1])*(1.0+2.0*exp(1im*k[1]*3.0/2.0)*cos(sqrt(3.0)*k[2]/2.0))
	H[1,2]=t_0*f_k
	H[2,1]=conj(H[1,2])
	return H
   end
   #
   #
   function Berry_Connection(k)  
        #
        #tau distance between neighbor
        tau=2.732
        #
	A=zeros(Complex{Float64},2,2,ndim)
        A[1,1,1]=0
        A[2,2,1]=tau/2.0
	A[2,2,2]=0.0
	A[1,1,2]=sqrt(3)/2.0*tau
	return A
   end
   #
   # Based on perturbation theory
   # Eq. 24 of https://arxiv.org/pdf/cond-mat/0608257.pdf
   #
   function Dipole_Matrices(k; dk=0.01)
       #
       #tau distance between neighbor
       tau=2.732
       #
       k_plus =copy(k)
       k_minus=copy(k)
       dH=zeros(Complex{Float64},2,2,ndim)
       #
       for id in 1:ndim
           k_plus[id] =k[id]+dk
           k_minus[id]=k[id]-dk
           H_plus =Hamiltonian(k_plus)
           H_minus=Hamiltonian(k_minus)
           dH[:,:,id]=(H_plus-H_minus)/(2.0*dk)*tau
           k_plus[id] =k[id]
           k_minus[id]=k[id]
       end
       return dH
   end
   #
   function Calculate_UdU(k, U;  dk=0.01)
       #
       #tau distance between neighbor
       tau=2.732
       #
       k_plus =copy(k)
       k_minus=copy(k)
       UdU=zeros(Complex{Float64},2,2,ndim)
       #
       for id in 1:ndim
           k_plus[id] =k[id]+dk
           k_minus[id]=k[id]-dk
           H_plus =Hamiltonian(k_plus)
	   data_plus= eigen(H_plus)
	   eigenvec_p = data_plus.vectors
	   eigenvec_p= fix_eigenvec_phase(eigenvec_p)

           H_minus=Hamiltonian(k_minus)
	   data_minus= eigen(H_minus)
	   eigenvec_m = data_minus.vectors
	   eigenvec_m= fix_eigenvec_phase(eigenvec_m)
	   
	   dU=(eigenvec_p-eigenvec_m)/(2.0*dk)*tau
	   UdU[:,:,id]=(U')*dU

           k_plus[id] =k[id]
           k_minus[id]=k[id]
       end
       #
       return UdU
   end
   #
end
