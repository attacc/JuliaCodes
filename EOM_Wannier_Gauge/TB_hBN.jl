#
# Module for Tight-binding code for monolayer hexagonal boron nitride
# Claudio Attaccalite (2023)
#

module hBN2D

using LinearAlgebra
include("TB_tools.jl")
using .TB_tools

include("units.jl")
using .Units
#
# In this program the lattice constant is equal to 1
#
#
# Default TB paramters from PRB 94, 125303 
t_0 = 2.30/ha2ev  # eV
E_gap=3.625*2.0/ha2ev  # eV
#
s_dim=2 # space dimension
h_dim=2 # hamiltonian dimension
#
# Distance between neighbor 
a_cc=2.632 # a.u.
# Lattice vectors:
a_1=a_cc/2.0*[3.0,  sqrt(3.0)]
a_2=a_cc/2.0*[3.0, -sqrt(3.0)]

export Hamiltonian,Berry_Connection,Grad_H,Calculate_UdU,a_1,a_2,s_dim,h_dim
  #
  global ndim=2
  #
  function Hamiltonian(k)::Matrix{Complex{Float64}}
        #
	H=zeros(Complex{Float64},2,2)
        #
        # Diagonal part 0,E_gap
        #
	H[1,1]= E_gap/2.0
	H[2,2]=-E_gap/2.0
        #
        # Off diagonal part
        # f(k)=e^{-i * k_y} * (1+2*e^{ i * k_y *3/2} ) * cos(sqrt(3)/2 *k_x)
        #
	f_k=exp(-1im*k[2]*a_cc)*(1.0+2.0*exp(1im*k[2]*3.0*a_cc/2.0)*cos(sqrt(3.0)*k[1]*a_cc/2.0))
	H[1,2]=t_0*f_k
	H[2,1]=conj(H[1,2])
	return H
   end
   #
   #
   function Berry_Connection(k)  
        #
        # Notice that in TB-approximation 
        # the Berry connect does not depend from k
        # but if we start from Wannier function it does
        #
	A=zeros(Complex{Float64},2,2,ndim)
        A[1,1,1]=0
        A[2,2,1]=2.0*a_cc
	A[1,1,2]=0.0 #sqrt(3)/2.0*a_cc
	A[2,2,2]=0.0
	return A
   end
   #
   # Based on perturbation theory
   # Eq. 24 of https://arxiv.org/pdf/cond-mat/0608257.pdf
   #
   function Grad_H(k; dk=0.01)
       #
       # calculate dH/dk in the Wannier Gauge
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
           dH[:,:,id]=(H_plus-H_minus)/(2.0*dk)
           k_plus[id] =k[id]
           k_minus[id]=k[id]
       end
       #
       return dH
   end
   #
   function k_deriv_to_cart(M_crys,lattice)
	M_cart=zeros(M_crys)
        for iv in lattice.dim,iv in lattice.dim
	  for id in lattice.dim
      	     M_cart[:,:,id]=M_cart[:,:,id]+lattice.vectors[iv][id]*M_crys[:,:,iv]*lattice.rb_norm[iv]
	  end
	end
        M_cart=M_cart/(2.0*pi)
      return M_cart
   end 
   #
   # Calculate derivatives along the rvectors directions
   #
   function Calculate_UdU(k, U;  dk=0.01)
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
	   
	   dU=(eigenvec_p-eigenvec_m)/(2.0*dk)
	   UdU[:,:,id]=(U')*dU

           k_plus[id] =k[id]
           k_minus[id]=k[id]
       end
       #
       return UdU
   end
   #
end
