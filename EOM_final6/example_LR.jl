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

include("TB_tools.jl")
using .TB_tools

include("bz_sampling.jl")
using .BZ_sampling

include("external_field.jl")
using .ExternalField

include("lattice.jl")
using .LatticeTools

include("Dipoles.jl")
include("Linear_response.jl")
# 
# Input parameters
#===================#
# 
# Create lattice using vectors defined in TB_hBN.jl
lattice=set_Lattice(2,[a_1,a_2])

# k_grid
n_k1=120
n_k2=120

# TB-gauge
TB_gauge=TB_atomic

# electric field
e_field.E_vec   = [0.0,1.0]

# k-derivatives
dk=0.01        # recalculate H(k+dk)
# dk=nothing   # use the MK grid

# Dipole calculation
use_GradH=true #false

# Spectrum range
freqs_range  =[0.0/ha2ev, 25.0/ha2ev] # eV
eta          =0.1/ha2ev
freqs_nsteps =1200
#
# Generate the Monkhorst-Pack grid
#
k_grid=generate_unif_grid(n_k1, n_k2, lattice)
#
# Solve the Hamiltonian on the grid and store the solution in TB_sol
#
TB_sol=Solve_TB_on_grid(k_grid,BN_Hamiltonian,TB_gauge)
#
# Calculate Dipoles
#
Dip_h,∇H_w=Build_Dipole(k_grid,lattice,TB_sol,TB_gauge,BN_orbitals,BN_Hamiltonian,dk,use_GradH)
#
# Frequenciy rage
#
freqs=LinRange(freqs_range[1],freqs_range[2],freqs_nsteps)
#
# Calculate Xhi
#
xhi = Linear_response(TB_sol, Dip_h, freqs, e_field.E_vec, eta)
#
# Plot result
#
fig = figure("Linear response",figsize=(10,20))
plot(freqs*ha2ev,real(xhi[:]))
plot(freqs*ha2ev,imag(xhi[:]))
PyPlot.show();

# Write on disk

function generate_header(k_grid,eta,Efield_ver,freqs)
    header="#\n# * * * Linear response xhi(ω) * * * \n#\n" 
    header*="# k-point grid: $(k_grid.nk_dir[1]) - $(k_grid.nk_dir[2]) \n"
    header*="# E-field versor: $(Efield_ver) \n"
    header*="# Frequencies range: $(freqs[1]*ha2ev) - $(freqs[end]*ha2ev) [eV]  \n"
    header*="# Frequencies steps: $(length(freqs)) \n#\n"
    return header
end

f = open("xhi_w.csv","w")
header=generate_header(k_grid,eta,e_field.E_vec,freqs)
write(f,header)
for iw in 1:freqs_nsteps
    write(f," $(freqs[iw]*ha2ev) $(imag(xhi[iw])) $(real(xhi[iw])) \n")
end         
close(f)
