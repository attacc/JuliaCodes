export plot_bands

using Plots

"""
    plot_bands(vals::AbstractMatrix; kwargs...)

Plot the band structure given an array of energy eigenvalues.

# Arguments
- `vals::AbstractMatrix`: A `(num_wann, n_k)` matrix containing the energy eigenvalues for each k-point 
                          (e.g., as returned by `get_bands`).
- `kwargs...`: Additional keyword arguments to pass to `Plots.plot`.

# Returns
A `Plots.Plot` object displaying the band structure.
"""
function plot_bands(vals::AbstractMatrix; kwargs...)
    # Determine the number of bands and k-points
    num_wann, n_k = size(vals)
    
    # We plot the eigenvalues. The x-axis is just the k-point index.
    # The `vals` matrix is plotted such that each row (band) is a distinct curve.
    # `plot` directly on a matrix plots the columns as series by default, 
    # so we transpose `vals` (`vals'`) to plot rows as series.
    p = plot(vals'; 
             xlabel="k-path index", 
             ylabel="Energy", 
             legend=false, 
             title="Band Structure",
             linewidth=2.0,
             kwargs...)
             
    return p
end
