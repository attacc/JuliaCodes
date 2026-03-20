using TightBinding

# In this example we load a 2D Boron Nitride Tight-Binding model, 
# which is represented by a Wannier90 hr.dat file.
# The BN model uses two Wannier functions per unit cell. 
# There is an on-site energy of M = +1.0 for Boron, and -1.0 for Nitrogen.
# The nearest neighbor hopping `t` is 2.7.

println("Loading 2D Boron Nitride model from hr.dat...")
model = read_tb_model(joinpath(@__DIR__, "boron_nitride_hr.dat"))

# Calculate the eigenvalues exactly at the Dirac point (K-point)
# For the standard orientation of our 2-band BN model, the K point is at [1/3, 1/3, 0]
# in fractional reciprocal coordinates.
K_point = [1/3, 1/3, 0.0]
println("Calculating Hamiltonian at the K-point (1/3, 1/3, 0)...")
H_K = hamiltonian(model, K_point)

display(H_K)

# Get the bands (eigenvalues) at the K point
vals_K = get_bands(model, [K_point])
println("\nBands at K-point (energy gap = $($(vals_K[2,1] - vals_K[1,1]))):")
println(vals_K)

# Calculate a quick band structure from Gamma [0,0,0] to K [1/3, 1/3, 0]
kpoints_path = [[k, k, 0.0] for k in range(0.0, 1/3, length=20)]
vals_path = get_bands(model, kpoints_path)

println("\nBand gap from Gamma to K:")
println("k = [0, 0, 0]: ", vals_path[:, 1])
println("...          ...")
println("k = [1/3, 1/3, 0]: ", vals_path[:, end])
