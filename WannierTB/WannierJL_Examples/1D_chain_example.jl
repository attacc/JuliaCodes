using Wannier
using WannierIO

# This script demonstrates how to build a 1D Chain Tight-Binding Model directly 
# using the Wannier.jl package rather than our custom TightBinding module.

println("Loading 1D Chain using Wannier.jl...")

# 1. Read the tight-binding HR matrix using WannierIO 
# (which is re-exported/used natively by Wannier.jl)
# We pull the hr.dat file from the tests folder.
file_path = joinpath(@__DIR__, "..", "test", "test_hr.dat")
Rvecs, degeneracies, H_R, header = read_w90_hrdat(file_path)

# 2. Construct the Hamiltonian
# Depending on your exact Wannier.jl version, you build the Hamiltonian 
# from the extracted Wannier real-space matrices.
H_1D = Hamiltonian(Rvecs, degeneracies, H_R)

# 3. Define the K-path 
# For a 1D chain, the Brillouin zone goes from 0 to 0.5 (in fractional coordinates)
k_path = [[k, 0.0, 0.0] for k in range(0.0, 0.5, length=100)]

# 4. Compute the band structure
evals = get_bands(H_1D, k_path)

println("1D Chain Band Energy at Gamma (k=0.0): ", evals[:, 1])
println("1D Chain Band Energy at Edge (k=0.5): ", evals[:, end])
