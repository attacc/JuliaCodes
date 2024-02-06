#
# Density matrix EOM in the Wannier Gauge (TB approximation)
# Claudio Attaccalite (2023)
#
using LinearAlgebra
using CSV
using DataFrames
using Base.Threads
using PyPlot


include("TB_hBN.jl")
using .hBN2D

include("TB_tools.jl")
using .TB_tools

include("units.jl")
using .Units

include("lattice.jl")
using .LatticeTools

include("bz_sampling.jl")
using .BZ_sampling

lattice=set_Lattice(2,[a_1,a_2])
oribtals=
# 
# Code This code is in Hamiltonian space
# in dipole approximation only at the K-point
#
# * * * DIPOLES * * * #
#
# if use_Dipoles=true  dipoles are calculated
# using dH/dh
#
# if use_Dipoles=false dipoles are calculated
# uding UdU with fixed phase
#
use_Dipoles=true


# a generic off-diagonal matrix example (0 1; 1 0)
off_diag=.~I(h_dim)

# K-points
n_k1=128
n_k2=1

k_grid=generate_unif_grid(n_k1, n_k2, lattice)

nk=n_k1*n_k2
#nk=1
#k_vec=[1,1/sqrt(3)]
#k_vec=k_vec*2*pi/(3*a_cc)
#k_list=k_vec
TB_sol.h_dim=2
TB_sol.eigenval=zeros(Float64,h_dim,nk)
TB_sol.eigenvec=zeros(Complex{Float64},h_dim,h_dim,nk)
TB_sol.H_w     =zeros(Complex{Float64},h_dim,h_dim,nk)

println(" K-point list ")
println(" nk = ",nk)
print_k_grid(k_grid, lattice)

println("Building Hamiltonian: ")

#TB_gauge=TB_lattice
TB_gauge=TB_atomic
dk=0.01

#Threads.@threads for ik in ProgressBar(1:nk)
for ik in 1:nk
   TB_sol.H_w[:,:,ik]=Hamiltonian(k_grid.kpt[:,ik],gauge=TB_gauge)
   #
   data= eigen(TB_sol.H_w[:,:,ik])      # Diagonalize the matrix
   TB_sol.eigenval[:,ik]   = data.values
   TB_sol.eigenvec[:,:,ik] = data.vectors
   TB_sol.eigenvec[:,:,ik] = fix_eigenvec_phase(TB_sol.eigenvec[:,:,ik],ik,k_grid)
end
#
# Write Hamiltonian on disk
#
h_file=open("hamiltonian_k.dat","w")
vec_file=open("eigenvec_k.dat","w")
for ik in 1:nk
    write(h_file," $(real(TB_sol.H_w[2,1,ik])) $(imag(TB_sol.H_w[2,1,ik])) ")
    write(h_file," $(TB_sol.eigenval[1,ik]) $(TB_sol.eigenval[2,ik]) \n")
    write(vec_file," $(real(TB_sol.eigenvec[1,1,ik])) $(imag(TB_sol.eigenvec[1,1,ik])) $(real(TB_sol.eigenvec[2,1,ik])) $(imag(TB_sol.eigenvec[2,1,ik]))  \n")
#    write(vec_file," $(real(TB_sol.eigenvec[1,2,ik])) $(imag(TB_sol.eigenvec[1,2,ik])) $(real(TB_sol.eigenvec[2,2,ik])) $(imag(TB_sol.eigenvec[2,2,ik]))  \n")
end
close(vec_file)
close(h_file)
# 
#Print Hamiltonian info
#
dir_gap=minimum(TB_sol.eigenval[2,:]-TB_sol.eigenval[1,:])
println("Direct gap : ",dir_gap*ha2ev," [eV] ")
ind_gap=minimum(TB_sol.eigenval[2,:])-maximum(TB_sol.eigenval[1,:])
println("Indirect gap : ",ind_gap*ha2ev," [eV] ")

#fig = figure("Eigenvalues",figsize=(10,20))
#plot(real(TB_sol.eigenvec[1,1,:]),label="Re[V_1[1]]") 
#plot(imag(TB_sol.eigenvec[1,1,:]),label="Im[V_1[1]]") 
#plot(real(TB_sol.eigenvec[2,1,:]),label="Re[V_1[2]]") 
#plot(imag(TB_sol.eigenvec[2,1,:]),label="Re[V_1[2]]") 
#legend()
#PyPlot.show();

# Dipoles
∇H_w=zeros(Complex{Float64},h_dim,h_dim,s_dim,nk)
Threads.@threads for ik in ProgressBar(1:nk)
  ∇H_w[:,:,:,ik]=Grad_H(ik,k_grid,lattice,Hamiltonian,TB_sol,TB_gauge,dk)
end

fig = figure("Grad H $TB_gauge",figsize=(10,20))
plot(real(real(∇H_w[1,2,1,:])),label="∇H_w[1,2,id=1]") 
plot(real(imag(∇H_w[1,2,1,:])),label="∇H_w[1,2,id=1]") 
#plot(real(real(∇H_w[2,1,2,:])),label="∇H_w[1,2,id=1]") 
#plot(real(imag(∇H_w[2,1,2,:])),label="∇H_w[1,2,id=1]") 
legend()
PyPlot.show();

