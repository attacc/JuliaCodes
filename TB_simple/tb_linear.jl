#
# Simple Tight-binding code for one-dimensional system
# Claudio Attaccalite (2024)
#
using Printf
using LinearAlgebra
using Plots
using Bessels

# In this program the lattice constant is equal to 1

function Hamiltonian(k; t_0 = 1.0/2.0, Q=0.0)::Matrix{Complex{Float64}}
        #
	H=zeros(Complex{Float64},2,2)
        #
        # Diagonal part 0,E_gap
        #
	H[1,1]=-Q
	H[2,2]=Q
        #
        # Off diagonal part
        #
        H[1,2]=t_0*cos(k[1])
	H[2,1]=conj(H[1,2])
	return H
end

function Floquet_Hamiltonian(k, F_modes; t_0 = 1.0/2.0, Q=0.0, omega=1.0, F=0.0)
        h_size =2
        n_modes=length(F_modes)
        H_flq=zeros(Complex{Float64},n_modes,n_modes,h_size,h_size)

#Diagonal terms respect to the mode and Q
        for i1 in 1:n_modes
          i_m=F_modes[i1]
          for ih in (1:h_size)
            H_flq[i1,i1,ih,ih]=i_m*omega+(-Q)^ih
          end
        end
        
#Off-diagonal terms in mode and t_0
        for i1 in 1:n_modes
          i_m=F_modes[i1]
          for i2 in 1:n_modes
             i_n=F_modes[i2]
             H_flq[i1,i2,1,2]=(1.0im)^(i_m-i_n)*besselj(i_m-i_n,F)*t_0*cos(k[1]-(i_m-i_n)*pi/2.0)
             H_flq[i1,i2,2,1]=H_flq[i1,i2,1,2]
          end
       end
       return copy(reshape(permutedims(H_flq,(1,3,2,4)),(n_modes*h_size,n_modes*h_size)))
end



println("")
println(" * * * Tight-binding code for 1D-system  * * *")
println("")
# 
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

L=1.0
zero=[0.0]
Pi2=[+pi/(2.0*L)]

points=[zero,Pi2]
n_kpt=30

t_0=0.5
Q=0.1

path=generate_circuit(points,n_kpt)
band_structure = zeros(Float64, length(path), 2)

for (i,kpt) in enumerate(path)
	H=Hamiltonian(kpt;t_0,Q)
	eigenvalues = eigen(H).values       # Diagonalize the matrix
        band_structure[i, :] = eigenvalues  # Store eigenvalues in an array
end
display(plot(band_structure[:, 1], label="Band 1", xlabel="k", ylabel="Energy [eV]", legend=:topright))
display(plot!(band_structure[:, 2], label="Band 2"))
title!("Band structure for two site 1D model")
sleep(5)

#
# Build the Floquet Hamiltonian
#
t_0=0.5     # hopping
Q=0.1       # gap
F=0.0       # Intensity
omega=1.0   # Frequency
max_mode=0  # max number of modes
h_size=2    # Hamiltonian size
#
F_modes=range(-max_mode,max_mode,step=1)
n_modes=length(F_modes)
#
@printf("Floquet Hamiltonian Q=%f  F=%f  max_mode=%d ",Q,F,max_mode)
#
flq_bands = zeros(Float64, length(path), n_modes*h_size)
for (i,kpt) in enumerate(path)
	H_flq=Floquet_Hamiltonian(kpt,F_modes;t_0,Q,omega,F)
	eigenvalues = eigen(H_flq).values       # Diagonalize the matrix
        flq_bands[i, :] = eigenvalues  # Store eigenvalues in an array
end
display(plot(flq_bands[:, 1], label="Band 1", xlabel="k", ylabel="Energy [eV]", legend=:topright))
display(plot!(flq_bands[:, 2], label="Band 2"))
#display(plot!(flq_bands[:, 3], label="Band 3"))
#display(plot!(flq_bands[:, 4], label="Band 4"))
title!("Floquet band structure for two site 1D model")
sleep(10)

