export TBModel, read_tb_model

"""
    TBModel

A structure to hold tight binding model parameters, typically read from a Wannier90 `hr.dat` file.

# Fields
- `num_wann::Int`: Number of Wannier functions (or bands) in the model.
- `nrpts::Int`: Number of Wigner-Seitz grid points (real-space lattice vectors `R`).
- `degeneracies::Vector{Int}`: Degeneracy weight of each `R` vector point.
- `Rvecs::Vector{<:AbstractVector{Int}}`: Array of real-space lattice vectors `R`.
- `H_R::Vector{Matrix{ComplexF64}}`: Array of Hamiltonian matrices in the Wannier basis, evaluated at each `R`.
"""
struct TBModel
    num_wann::Int
    nrpts::Int
    degeneracies::Vector{Int}
    Rvecs::Vector{<:AbstractVector{Int}}
    H_R::Vector{Matrix{ComplexF64}}
end

"""
    read_tb_model(filename::String) -> TBModel

Read a tight binding model from a Wannier90 `hr.dat` file.

# Arguments
- `filename::String`: Path to the `hr.dat` file.

# Returns
- A `TBModel` instance containing the parsed real-space Hamiltonian matrix elements.
"""
function read_tb_model(filename::String)
    # Parse data from the hr.dat file using WannierIO
    Rvecs, degeneracies, H_R, header = WannierIO.read_w90_hrdat(filename)
    
    # Infer the number of Wannier functions from the size of the first Hamiltonian matrix
    num_wann = size(H_R[1], 1)
    
    # The number of R points matches the length of the degeneracies array
    nrpts = length(degeneracies)
    
    return TBModel(num_wann, nrpts, degeneracies, Rvecs, H_R)
end
