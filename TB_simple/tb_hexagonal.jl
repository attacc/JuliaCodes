#
# Simple Tight-binding code for monolayer hexagonal boron nitride
# Claudio Attaccalite (2023)
#
using Printf
using LinearAlgebra
using Plots

#=
 Lattice vectors
 
 a_1=a/2 (1,  sqrt(3))
 a_2=a  ( 1, 0 )
 
 First neighboars
 
 delta_1 =  a/2 ( 1 ,  sqrt(3))
 delta_2 =  a/2 ( 1 , -sqrt(3))
 delta_3 =  a/2 (-2 , 0 )
 
 Volume = a_1 * a_2 = 3/2 a^2


 Reciprocal space
 
 b_1 = 2\pi * 2/3a (1,  1/sqrt(3)) 
 b_2 = 2\pi * 2/3a (1, -1/sqrt(3))

 Special K-points

 K =  2\pi/3a (1,  1/sqrt(3))  
 K'= -2\pi/3a (1, -1/sqrt(3))

 

 M = b_1/2
 M'= b_2/2

=# 

# In this program the lattice constant is equal to 1

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

println("")
println(" * * * Tight-binding code for hBN monolayer  * * *")
println("")
# 
# Define a circuit in k-space
# Gamma -> K -> M -> Gamma
#
function generate_circuit(points, n_steps)
	println("Generate k-path ")
	n_points=length(points)
	@printf("number of points:%5i \n",n_points)
	if n_points <= 1
		error("number of points less or equal to 1 ")
	end
	for i in 1:n_points
		@printf("v(%d) = %s \n",i,points[i])
	end
	path=Any[]
	for i in 1:(n_points-1)
		for j in 0:(n_steps-1)	
			dp=(points[i+1]-points[i])/n_steps
			push!(path,points[i]+dp*j)
		end
	end
	push!(path,points[n_points])
	return path
end

K=[1,1/sqrt(3)]
K=K*2*pi/3

M=[1.0/2.0,sqrt(3)/2.0]
M=M*2*pi/3

Gamma=[0,0]

points=[Gamma,K,M,Gamma]

path=generate_circuit(points,10)
band_structure = zeros(Float64, length(path), 2)

for (i,kpt) in enumerate(path)
	H=Hamiltonian(kpt)
	eigenvalues = eigen(H).values       # Diagonalize the matrix
        band_structure[i, :] = eigenvalues  # Store eigenvalues in an array
end
display(plot(band_structure[:, 1], label="Band 1", xlabel="k", ylabel="Energy [eV]", legend=:topright))
display(plot!(band_structure[:, 2], label="Band 2"))
title!("Band structure for a model 2D-hBN")
sleep(10)
