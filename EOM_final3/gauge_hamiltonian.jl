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

n_fft_k1=10
n_fft_k2=1

k_grid_fft=generate_unif_grid(n_fft_k1, n_fft_k2, lattice)

nk_fft=n_fft_k1*n_fft_k2
H_w     =zeros(Complex{Float64},h_dim,h_dim,nk_fft)

println(" K-point list for the FFT")
println(" nk = ",nk_fft)
print_k_grid(k_grid_fft, lattice)

println("Building Hamiltonian: ")
Threads.@threads for ik in ProgressBar(1:nk_fft)
   H_w[:,:,ik]=Hamiltonian(k_grid_fft.kpt[:,ik])
end
#
r_grid=generate_R_grid(lattice,k_grid_fft)
#
H_R=zeros(Complex{Float64},h_dim,h_dim,nk_fft)

# Build H_R
println("Building H_R ")
Threads.@threads for iR in ProgressBar(1:nk_fft)
  for ik in 1:nk_fft
    for i1 in 1:h_dim,i2 in 1:h_dim
        H_R[i1,i2,iR]+=exp(1im*dot(r_grid.R_vec[:,iR],k_grid_fft.kpt[:,ik]))*H_w[i1,i2,ik]
   end
  end
end
#

function get_Hk_from_Hr(H_R,r_grid,k_grid,ik)
 H_k=zeros(Complex{Float64},h_dim,h_dim)
 for iR in 1:r_grid.nR
   for i1 in 1:h_dim
     for i2 in i1:h_dim
       H_k[i1,i2]+=exp(-1im*dot(r_grid.R_vec[:,iR],k_grid.kpt[:,ik]))*H_R[i1,i2,iR]
       H_k[i2,i1]=conj(H_k[i1,i2])
     end
     H_k[i1,i1]=real(H_k[i1,i1])
     end
  end
  return H_k
end


n_k1=128
n_k2=1
nk=n_k1*n_k2
k_grid=generate_unif_grid(n_k1, n_k2, lattice)
TB_sol.h_dim=2
TB_sol.eigenval=zeros(Float64,h_dim,nk)
TB_sol.eigenvec=zeros(Complex{Float64},h_dim,h_dim,nk)
TB_sol.H_w     =zeros(Complex{Float64},h_dim,h_dim,nk)
#


println("Building Hamiltonian from H_R: ")
for ik in ProgressBar(1:nk)
   TB_sol.H_w[:,:,ik]=get_Hk_from_Hr(H_R,r_grid,k_grid,ik)
   data= eigen(TB_sol.H_w[:,:,ik])      # Diagonalize the matrix
   TB_sol.eigenval[:,ik]   = data.values
   TB_sol.eigenvec[:,:,ik] = data.vectors
   TB_sol.eigenvec[:,:,ik] = fix_eigenvec_phase(TB_sol.eigenvec[:,:,ik],ik,k_grid)
end

h_file=open("hamiltonian_fft_k.dat","w")
vec_file=open("eigenvec_fft_k.dat","w")
for ik in 1:nk
    write(h_file," $(real(TB_sol.H_w[2,1,ik])) $(imag(TB_sol.H_w[2,1,ik])) ")
    write(h_file," $(TB_sol.eigenval[1,ik]) $(TB_sol.eigenval[2,ik]) \n")
    write(vec_file," $(real(TB_sol.eigenvec[1,1,ik])) $(imag(TB_sol.eigenvec[1,1,ik])) $(real(TB_sol.eigenvec[2,1,ik])) $(imag(TB_sol.eigenvec[2,1,ik]))  \n")
#    write(vec_file," $(real(TB_sol.eigenvec[1,2,ik])) $(imag(TB_sol.eigenvec[1,2,ik])) $(real(TB_sol.eigenvec[2,2,ik])) $(imag(TB_sol.eigenvec[2,2,ik]))  \n")
end
close(vec_file)
close(h_file)
#
