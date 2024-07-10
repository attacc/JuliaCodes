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

include("external_field.jl")
using .ExternalField

include("Dipoles.jl")
include("Linear_response.jl")

lattice =set_Lattice(2,[a_1,a_2])
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
use_GradH=true

# a generic off-diagonal matrix example (0 1; 1 0)
off_diag=.~I(h_dim)

lattice=set_Lattice(2,[a_1,a_2])

n_k1=64
n_k2=64

#
# Gauge for the tight-binding
#
#TB_gauge=TB_lattice
TB_gauge=TB_atomic
#
# Step for finite differences in k-space
#
# dk=nothing 
dk=0.01

# For Linear reponse only
freqs_range  =[0.0/ha2ev, 25.0/ha2ev] # eV
eta          =0.15/ha2ev
freqs_nsteps =400
e_field.E_vec   = [0.0,1.0]

k_grid=generate_unif_grid(n_k1, n_k2, lattice)
# 
# Solve TB on a regular grid
#
TB_sol=Solve_TB_on_grid(k_grid,Hamiltonian,TB_gauge)
# 
# Dipoles
#
Dip_h,∇H_w=Build_Dipole(k_grid,lattice,TB_sol,TB_gauge,orbitals,Hamiltonian,dk,use_GradH)
#
freqs=LinRange(freqs_range[1],freqs_range[2],freqs_nsteps)
#
# Calculate Xhi
#
xhi = Linear_response(TB_sol, Dip_h, freqs, e_field.E_vec, eta)
#
# Plot and write
#
function generate_header(k_grid,eta,Efield_ver,freqs)
    header="#\n# * * * Linear response xhi(ω) * * * \n#\n" 
    header*="# k-point grid: $(k_grid.nk_dir[1]) - $(k_grid.nk_dir[2]) \n"
    header*="# E-field versor: $(Efield_ver) \n"
    header*="# Frequencies range: $(freqs[1]*ha2ev) - $(freqs[end]*ha2ev) [eV]  \n"
    header*="# Frequencies steps: $(length(freqs)) \n#\n"
    return header
end
# 
# Plot and write on disk
#
fig = figure("Linear response",figsize=(10,20))
plot(freqs*ha2ev,real(xhi[:]))
plot(freqs*ha2ev,imag(xhi[:]))
PyPlot.show();


f = open("xhi_w.csv","w")
header=generate_header(k_grid,eta,e_field.E_vec,freqs)
write(f,header)
for iw in 1:freqs_nsteps
    write(f," $(freqs[iw]*ha2ev) $(imag(xhi[iw])) $(real(xhi[iw])) \n")
end         
close(f)
