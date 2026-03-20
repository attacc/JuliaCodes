export hamiltonian

"""
    hamiltonian(model::TBModel, k::AbstractVector) -> Matrix{ComplexF64}

Evaluate the k-space Hamiltonian `H(k)` for a given tight-binding model at k-point `k`.

# Arguments
- `model::TBModel`: The tight-binding model containing real-space hopping parameters.
- `k::AbstractVector`: The k-point at which to evaluate the Hamiltonian.
  Expected to be in fractional (reciprocal lattice) coordinates.

# Returns
- A `num_wann × num_wann` Hermitian matrix representing the Hamiltonian at `k`.
"""
function hamiltonian(model::TBModel, k::AbstractVector)
    num_wann = model.num_wann
    nrpts = model.nrpts
    
    # Initialize the momentum-space Hamiltonian with zeros
    H_k = zeros(ComplexF64, num_wann, num_wann)
    
    for ir in 1:nrpts
        # R is a 3-element vector of integers representing the real-space lattice vector
        R = model.Rvecs[ir]
        
        # dot product k . R in fractional coordinates computes the Bloch phase
        phase = 2π * dot(k, R)
        
        # Factor includes the phase and the degeneracy weight of the Wigner-Seitz point
        factor = exp(im * phase) / model.degeneracies[ir]
        
        # Accumulate the contribution from each R point to the momentum-space Hamiltonian
        @. H_k += factor * model.H_R[ir]
    end
    
    # Ensure exact Hermiticity to avoid issues with degenerate eigenvalues.
    # Sometimes due to numerical precision the output might be slightly non-Hermitian.
    return (H_k + H_k') / 2
end
