#
# Density matrix EOM in the Wannier Gauge (TB approximation)
# Claudio Attaccalite (2023)
#
using LinearAlgebra
using DataFrames
using Base.Threads
using PyPlot

include("units.jl")
using .Units

include("TB_hBN.jl")
using .hBN2D

include("lattice.jl")
using .LatticeTools

include("TB_tools.jl")
using .TB_tools

include("bz_sampling.jl")
using .BZ_sampling

include("Dipoles.jl")
# 
# Code This code is in Hamiltonian space
# in dipole approximation only at the K-point
#
# * * * DIPOLES * * * #
#
# if use_gradH=true  dipoles are calculated
# using dH/dh
#
# if use_GradH=false dipoles are calculated
# uding UdU with fixed phase
#
#use_GradH=false
use_GradH=true

# a generic off-diagonal matrix example (0 1; 1 0)
off_diag=.~I(h_dim)

lattice=set_Lattice(2,[a_1,a_2])

n_k1=2
n_k2=2

#
# Gauge for the tight-binding
#
TB_gauge=TB_lattice
#TB_gauge=TB_atomic   
#
# Step for finite differences in k-space
#
# dk=nothing 
dk=0.001

# For Linear reponse only
freqs_range  =[0.0/ha2ev, 25.0/ha2ev] # eV
eta          =0.15/ha2ev
freqs_nsteps =400
#
k_grid=generate_unif_grid(n_k1, n_k2, lattice)
#
# Write k-points on file
println("\n\n * * * K-points grid  * * * \n\n")
#
print_k_grid(k_grid,lattice)
#
# 
# Solve TB on a regular grid
#
TB_sol=Solve_TB_on_grid(k_grid,Hamiltonian,TB_gauge)
# 
# Print Hamiltonian
#
println("\n\n\n* * * Matrix elements of the Hamiltonian [eV] * * * ")
for ik in 1:k_grid.nk
    println(" ik = $ik ")
    println("H(1,1) = $(TB_sol.H_w[1,1,ik]*ha2ev)  [eV]")
    println("H(1,2) = $(TB_sol.H_w[1,2,ik]*ha2ev)  [eV]")
end
#
# Dipoles
#
Dip_h,∇H_w=Build_Dipole(k_grid,lattice,TB_sol,TB_gauge,orbitals,Hamiltonian,dk,use_GradH)
println("\n\n\n* * * Gradient of H (wannier gauge) [Atomic Units]* * * ")
for ik in 1:k_grid.nk
    println(" ik = $ik ")
    println("∇H_x(1,1) = $(∇H_w[1,1,1,ik])        ∇H_y(1,1) = $(∇H_w[1,1,2,ik])")
    println("∇H_x(1,2) = $(∇H_w[1,2,1,ik])        ∇H_y(1,2) = $(∇H_w[1,2,2,ik])")
end
println("\n\n\n* * * Dipole U'∇HU/(ϵ_i-ϵ_j) [Atomic units] * * * ")
for ik in 1:k_grid.nk
    println(" ik = $ik ")
    println("Dip^h_x(1,2) = $(Dip_h[1,2,1,ik])     Dip^h_y(1,2) = $(Dip_h[1,2,2,ik])")
    e_ij=TB_sol.eigenval[2,ik]-TB_sol.eigenval[1,ik]
    println("ϵ_i-ϵ_j = $(e_ij)")
    println("<u_i| ∇H^h_x| u_j> = $(Dip_h[1,2,1,ik]*e_ij)")
end

#
