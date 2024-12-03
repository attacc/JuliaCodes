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
        H[1,2]=t_0*(exp(-1im*k[1])+exp(+1im*k[1]))
	H[2,1]=conj(H[1,2])
#        println(H[1,2])
	return H
end

function Floquet_Hamiltonian(k; t_0 = 1.0/2.0, Q=0.0, omega=1.0, F=0.0, n_modes=2)::Matrix{Complex{Float64}}
        h_size =2
        H_flq_size=n_modes*h_size
        
        H_flq=zeros(Complex{Float64},H_flq_size,H_flq_size)

#Diagonal terms respect to mode and Q
        for i_m in (1:n_modes)
          for ih in (1:h_size)
            H_flq[(i_m-1)*h_size+ih,(i_m-1)*h_size+ih]=i_m*omega+(-Q)^ih
          end
        end
        
#Off-diagonal terms in mode and t_0
        for i_m in (1:n_modes)
          for i_n in (i_m:n_modes)
              H_flq[(i_m-1)*h_size+1,(i_n-1)*h_size+2]=(1.0im)^(i_m-i_n)*besselj(i_m-i_n,F)*cos(k[1]-(i_m-i_n)*pi/2.0)
              H_flq[(i_m-1)*h_size+2,(i_n-1)*h_size+1]=H_flq[(i_m-1)*h_size+1,(i_n-1)*h_size+2]
              #
              # complex conjugate
              #
              H_flq[(i_n-1)*h_size+2,(i_m-1)*h_size+1]=conj(H_flq[(i_m-1)*h_size+2,(i_n-1)*h_size+1])
              H_flq[(i_n-1)*h_size+1,(i_m-1)*h_size+2]=conj(H_flq[(i_m-1)*h_size+1,(i_n-1)*h_size+2])
          end
       end
    return H_flq
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
mPi=[-pi/(2.0*L)]
pPi=[+pi/(2.0*L)]

points=[mPi,pPi]
n_kpt=10

t_0=0.5
Q=0.1

path=generate_circuit(points,n_kpt)
band_structure = zeros(Float64, length(path), 2)

for (i,kpt) in enumerate(path)
	H=Hamiltonian(kpt;t_0,Q)
	eigenvalues = eigen(H).values       # Diagonalize the matrix
        band_structure[i, :] = eigenvalues  # Store eigenvalues in an array
        println(eigenvalues[1])
end
display(plot(band_structure[:, 1], label="Band 1", xlabel="k", ylabel="Energy [eV]", legend=:topright))
display(plot!(band_structure[:, 2], label="Band 2"))
title!("Band structure for two site 1D model")
sleep(10)

#
# Build the Floquet Hamiltonian
#
t_0=0.5
Q=1.0
F=0.0
omega=1.0
n_modes=2
h_size=2
#
flq_bands = zeros(Float64, length(path), n_modes*h_size)
for (i,kpt) in enumerate(path)
	H_flq=Floquet_Hamiltonian(kpt;t_0,Q,omega,F,n_modes)
	eigenvalues = eigen(H_flq).values       # Diagonalize the matrix
        flq_bands[i, :] = eigenvalues  # Store eigenvalues in an array
end
display(plot(flq_bands[:, 1], label="Band 1", xlabel="k", ylabel="Energy [eV]", legend=:topright))
display(plot!(flq_bands[:, 2], label="Band 2"))
display(plot!(flq_bands[:, 3], label="Band 3"))
display(plot!(flq_bands[:, 4], label="Band 4"))
title!("Floquet band structure for two site 1D model")
sleep(10)

