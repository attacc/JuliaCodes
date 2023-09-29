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
 atom_2=(1, sqrt(3.0)

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

# In this program the lattice constant is equal to 1

export Hamiltonian,Berry_connection
  #
  function Hamiltonian(k; t_0 = 2.92, E_gap=5.62)::Matrix{Complex{Float64}}
        #
        # Default paramters from PRB 100, 195201 (2019)
        #
	H=zeros(Complex{Float64},2,2)
        #
        # Diagonal part 0,E_gap
        #
	H[1,1]=E_gap
	H[2,2]=0.0
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
   function Berry_Connection(k; tau=1.0)
        A_x[1,1]=tau
	A_x[2,1]=0
	A_x[1,2]=0
	A_x[2,2]=tau
	A_y[1,1]=sqrt(3)*tau
	A_y[2,1]=0
	A_y[1,2]=0
	A_y[2,2]=sqrt(3)*tau
	return A_x,A_y
   end
   #
end
