export get_bands

"""
    get_bands(model::TBModel, kpoints::AbstractVector; vectors::Bool=false)

Compute the band structure (eigenvalues) for a list of k-points.

# Arguments
- `model::TBModel`: The tight-binding model.
- `kpoints::AbstractVector`: A vector of k-points, where each k-point is an `AbstractVector`
  in fractional coordinates.
- `vectors::Bool=false`: If `true`, also return the eigenvectors along with the eigenvalues.

# Returns
- `vals`: A `(num_wann, n_k)` matrix containing the energy eigenvalues for each k-point.
- `vecs` (optional): A `(num_wann, num_wann, n_k)` tensor containing the eigenvectors
  (returned only if `vectors=true`).
"""
function get_bands(model::TBModel, kpoints::AbstractVector; vectors::Bool=false)
    n_k = length(kpoints)
    n_bands = model.num_wann
    
    # Preallocate an array for the eigenvalues
    vals = zeros(Float64, n_bands, n_k)
    
    # Conditionally preallocate an array for the eigenvectors if vectors are requested
    vecs = vectors ? zeros(ComplexF64, n_bands, n_bands, n_k) : nothing
    
    for (i, k) in enumerate(kpoints)
        # Construct the Hamiltonian matrix at the current k-point
        H_k = hamiltonian(model, k)
        
        if vectors
            # Compute both eigenvalues and eigenvectors
            F = eigen(H_k)
            vals[:, i] = real.(F.values)
            vecs[:, :, i] = F.vectors
        else
            # Compute only eigenvalues for efficiency
            vals[:, i] = real.(eigvals(H_k))
        end
    end
    
    if vectors
        return vals, vecs
    else
        return vals
    end
end
