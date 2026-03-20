using Wannier
using WannierIO

# This script demonstrates how to build a 2D Boron Nitride Tight-Binding Model 
# directly using the Wannier.jl package.

println("Loading 2D Boron Nitride using Wannier.jl...")

# 1. Read the tight-binding HR matrix using WannierIO
file_path = joinpath(@__DIR__, "..", "examples", "boron_nitride_hr.dat")
Rvecs, degeneracies, H_R, header = read_w90_hrdat(file_path)

# 2. Construct the Hamiltonian
H_BN = Hamiltonian(Rvecs, degeneracies, H_R)

# 3. Define the High-Symmetry K-points Path
# Gamma [0,0,0] -> K [1/3, 1/3, 0] -> M [0.5, 0.0, 0.0]
k_path = [
    [0.0, 0.0, 0.0],       # Gamma
    [1/3, 1/3, 0.0],       # K (Dirac point)
    [0.5, 0.0, 0.0]        # M
]

# 4. Compute the band energies at these symmetry points
evals = get_bands(H_BN, k_path)

println("Bandgap exactly at the K-point:")
# At K-point (index 2 in our path array), the bands are split by the staggered on-site energies
gap = evals[2, 2] - evals[1, 2] 
println("Gap = ", gap, " eV")
