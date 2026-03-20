"""
    TightBinding

A module for parsing tight-binding models (e.g., from Wannier90 hr.dat files),
evaluating the Hamiltonian in reciprocal space, and calculating band structures.
"""
module TightBinding

using WannierIO
using LinearAlgebra

include("model.jl")
include("hamiltonian.jl")
include("bands.jl")
include("plot.jl")
include("rotation.jl")
include("position.jl")

end
