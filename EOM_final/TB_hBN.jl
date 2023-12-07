#
# Module for Tight-binding code for monolayer hexagonal boron nitride
# Claudio Attaccalite (2023)
#
module hBN2D

using LinearAlgebra

include("lattice.jl")
using .LatticeTools

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

export Hamiltonian,Berry_Connection,Grad_H,Calculate_UdU,a_1,a_2,s_dim,h_dim,a_cc,Calculate_Berry_Conec_FD
  #
  global ndim=2
  #
  global off_diag=.~I(h_dim)
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
        # f(k)=e^{-i*k_y*a} * (1+2*e^{ i*k_y*3*a/2} ) * cos(sqrt(3)*a/2*k_x)
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
        A[2,2,1]=a_cc/2.0
	A[1,1,2]=0.0 #sqrt(3)/2.0*a_cc
	A[2,2,2]=sqrt(3.0)/2.0*a_cc
	return A
   end
   #
   # Based on perturbation theory
   # Eq. 24 of https://arxiv.org/pdf/cond-mat/0608257.pdf
   #
   function Grad_H(k;  dk=0.01)
       #
       # calculate dH/dk in the Wannier Gauge
       # derivatives are in cartesian coordinates
       #
       k_plus =copy(k)
       k_minus=copy(k)
       dH=zeros(Complex{Float64},2,2,ndim)
       #
       for iv in 1:ndim
	   k_plus     =copy(k)
	   k_minus    =copy(k)
	   k_plus[iv] =k[iv]+dk
	   k_minus[iv]=k[iv]-dk
           H_plus =Hamiltonian(k_plus)
           H_minus=Hamiltonian(k_minus)
           dH[:,:,iv]=(H_plus-H_minus)/(2.0*dk)
       end
       #
       return dH
   end
   #
   # Calculate derivatives along the rvectors directions
   #
   function Calculate_UdU(ik, k_grid, eigenvecs, lattice;  dk=0.01)
    #
    U=eigenvecs[:,:,ik]
    UdU=zeros(Complex{Float64},2,2,ndim)
    #
     for id in 1:s_dim
       #
       if k_grid.nk_dir[id]==1
         continue
       end
       #
       ik_plus =get_k_neighbor(ik,id, 1,k_grid)
       ik_minus=get_k_neighbor(ik,id,-1,k_grid)
       #
       dk=norm(k_grid.kpt[:,ik_plus]-k_grid.kpt[:,ik_minus])/2.0
       #
       eigenvec_p = eigenvecs[:,:,ik_plus]
       eigenvec_m = eigenvecs[:,:,ik_minus]
       #
       dU=(eigenvec_p-eigenvec_m)/(2.0*dk)
       UdU[:,:,id]=((U')*dU) 
       #
     end
     #
     UdU=K_crys_to_cart(UdU,lattice)
     return UdU
#
#       k_plus =copy(k)
#       k_minus=copy(k)
#       UdU=zeros(Complex{Float64},2,2,ndim)
#       #
#       for id in 1:ndim
#           k_plus[id] =k[id]+dk
#           k_minus[id]=k[id]-dk
#           H_plus =Hamiltonian(k_plus)
#	   data_plus= eigen(H_plus)
#	   eigenvec_p = data_plus.vectors
#	   eigenvec_p= fix_eigenvec_phase(eigenvec_p)
#
#           H_minus=Hamiltonian(k_minus)
#	   data_minus= eigen(H_minus)
#	   eigenvec_m = data_minus.vectors
#	   eigenvec_m= fix_eigenvec_phase(eigenvec_m)
#	   
#	   dU=(eigenvec_p-eigenvec_m)/(2.0*dk)
#	   # I remove the identity, parallel guage
#	   UdU[:,:,id]=((U')*dU) #.*off_diag
#
#           k_plus[id] =k[id]
#           k_minus[id]=k[id]
#       end
#       #
#       return UdU
#       k_plus =copy(k)
#       k_minus=copy(k)
#       UdU=zeros(Complex{Float64},2,2,ndim)
#       #
#       for id in 1:ndim
#           k_plus[id] =k[id]+dk
#           k_minus[id]=k[id]-dk
#           H_plus =Hamiltonian(k_plus)
#	   data_plus= eigen(H_plus)
#	   eigenvec_p = data_plus.vectors
#	   eigenvec_p= fix_eigenvec_phase(eigenvec_p)
#
#           H_minus=Hamiltonian(k_minus)
#	   data_minus= eigen(H_minus)
#	   eigenvec_m = data_minus.vectors
#	   eigenvec_m= fix_eigenvec_phase(eigenvec_m)
#	   
#	   dU=(eigenvec_p-eigenvec_m)/(2.0*dk)
#	   # I remove the identity, parallel guage
#	   UdU[:,:,id]=((U')*dU) #.*off_diag
#
#           k_plus[id] =k[id]
#           k_minus[id]=k[id]
#       end
#       #
#       return UdU
   end
   #
   function Calculate_Berry_Conec_FD(ik, k_grid, U; dk=0.01)
       #
       k_plus =copy(k_grid.kpt[:,ik])
       k_minus=copy(k_grid.kpt[:,ik])
       A_w=zeros(Complex{Float64},2,2,ndim)
       #
       for id in 1:ndim
	   if k_grid.nk_dir[id]==1
	      continue
	   end
           k_plus[id] =k_grid.kpt[id,ik]+dk
           k_minus[id]=k_grid.kpt[id,ik]-dk
           H_plus =Hamiltonian(k_plus)
	   data_plus= eigen(H_plus)
	   eigenvec_p = data_plus.vectors
	   eigenvec_p= fix_eigenvec_phase(eigenvec_p)

           H_minus=Hamiltonian(k_minus)
	   data_minus= eigen(H_minus)
	   eigenvec_m = data_minus.vectors
	   eigenvec_m= fix_eigenvec_phase(eigenvec_m)
	   
	   dS=(U*eigenvec_p-U*eigenvec_m)/(2.0*dk)
	   A_w[:,:,id]=1im*dS  
	   #Rotation to Hamiltonian gauge is performed outside

           k_plus[id] =k_grid.kpt[id,ik]
           k_minus[id]=k_grid.kpt[id,ik]
       end
       #
       return A_w
   end
   #
end
