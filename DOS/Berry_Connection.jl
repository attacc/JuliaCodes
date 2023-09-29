#
# Density matrix EOM in the Wannier Gauge (TB approximation)
# Claudio Attaccalite (2023)
#
using LinearAlgebra
using Plots

include("TB_hBN.jl")
using .hBN2D

include("TB_tools.jl")
using .TB_tools

# 
# Reciprocal Lattice Vectors
#
b_1=2.0*pi/3*[1.0, sqrt(3)]
b_2=2.0*pi/3*[1.0, -sqrt(3)]
#
n_kx=40
n_ky=40
#
k_grid=generate_unif_grid(n_kx,n_ky,b_1,b_2)

#kx=[k_vec[1] for k_vec in k_grid]
#ky=[k_vec[2] for k_vec in k_grid]
#display(scatter(kx,ky,label=""))
#sleep(30)

n_bands=2

bands = zeros(Float64, length(k_grid), n_bands)
for (i,kpt) in enumerate(k_grid)
        H=Hamiltonian(kpt)
        eigenvalues = eigen(H).values       # Diagonalize the matrix
        bands[i, :] = eigenvalues           # Store eigenvalues in an array
end


E_range=[-8.0,14.0]
n_points=200
smearing=0.4

DOS=evaluate_DOS(bands,E_range,n_points,smearing)

display(plot(DOS[:,1],DOS[:,2]))
sleep(30)



