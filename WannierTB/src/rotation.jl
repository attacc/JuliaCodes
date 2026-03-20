export read_rotation_matrices

"""
    read_rotation_matrices(filename::String) -> Array{ComplexF64, 3}

Read spatial/gauge rotation matrices from a Wannier90 format file 
(e.g., `_u.mat` or `_u_dis.mat`) using WannierIO.

# Arguments
- `filename::String`: The path to the unitary matrix file.

# Returns
- A `(num_wann, num_wann, nkpts)` 3D array of unitary rotation matrices.
"""
function read_rotation_matrices(filename::String)
    # WannierIO.read_w90_umat automatically parses the _u.mat files
    return WannierIO.read_w90_umat(filename)
end
