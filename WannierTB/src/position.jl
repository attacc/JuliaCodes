export read_position_matrices

"""
    read_position_matrices(filename::String) -> Array{ComplexF64, 4}

Read the position operator matrix elements from a Wannier90 format file 
(e.g., `_r.mat`) using WannierIO.

# Arguments
- `filename::String`: The path to the position matrix file.

# Returns
- A `(3, num_wann, num_wann, nkpts)` 4D array of the position operator matrix 
  elements `⟨u_mk | r | u_nk⟩` for the 3 Cartesian directions.
"""
function read_position_matrices(filename::String)
    # WannierIO.read_w90_rmat automatically parses the _r.mat files
    return WannierIO.read_w90_rmat(filename)
end
