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

# Atom positions
d_1=      [0.0,0.0]
d_2=-a_cc*[1.0,0.0]
#d_1= a_cc/2.0*[1.0,  sqrt(3.0)]
#d_2= a_cc/2.0*[1.0, -sqrt(3.0)]
#d_3=-a_cc*[1.0,0.0]
#d_4= [0.0,0.0]

TB_lattice="lattice"
TB_atomic ="atomic"

orbitals=set_Orbitals(2,[d_1,d_2])

export Hamiltonian,Berry_Connection,a_1,a_2,s_dim,h_dim,a_cc,orbitals,TB_lattice,TB_atomic
  #
  global ndim=2
  #
  global off_diag=.~I(h_dim)
  #
  function Hamiltonian(k; gauge=TB_lattice)::Matrix{Complex{Float64}}
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
	f_k=exp(-1im*k[1]*a_cc)*(1.0+2.0*exp(1im*k[1]*3.0*a_cc/2.0)*cos(sqrt(3.0)*k[2]*a_cc/2.0))
	H[1,2]=t_0*f_k
        #
        # Transform the Hamiltonian in "atomic gauge" see notes
        # the "atomic gauge" is equivalent to the periodic part
        # of the Bloch functions only, while the "lattice guage"
        # includes also the k-dependent phase factor 
        #
        if gauge==TB_atomic
          d_tau=orbitals.tau[2]-orbitals.tau[1]
          k_dot_dtau=dot(k,d_tau)
          H[1,2]=H[1,2]*exp(-1im*k_dot_dtau)
        end
        # 
	H[2,1]=conj(H[1,2])
        #
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
        # I choose the atom 1 at the centrer of the axis
        # therefore in the cell I have only a second atom at t_n=-d_3
        # all the others are connected by a vector R but \delta_{R,0}
        #
        # Berry Connection is nothing else the the gradient
        # of the similarity transformation
        # VdV see TB_tools.jl
        #
	A=zeros(Complex{Float64},2,2,ndim)
        for ih in 1:h_dim
          A[ih,ih,:]=-1im*orbitals.tau[ih][:]
        end
	return A
   end
   #
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
