using LinearAlgebra
using SpecialFunctions
using PyPlot
using ProgressBars

# Define Pauli matrices
const σ0 = [1 0; 0 1]
const σ1 = [0 1; 1 0]
const σ3 = [1 0; 0 -1]

# Global parameters (will be set later)
Nm = 3
W = 1.0
F = 0.5
Q = 0.1
Nk = 50

# Hamiltonian matrix construction
function H(k::Float64)
    n_range = -Nm:Nm
    m_range = -Nm:Nm
    
    # Build the diagonal block matrix
    blocks = []
    for n in n_range
        row_blocks = []
        for m in m_range
            if n == m
                block = (n * W * σ0 + Q * σ3)
            else
                block = zeros(2, 2)
            end
            push!(row_blocks, block)
        end
        push!(blocks, hcat(row_blocks...))
    end
    
    H0 = vcat(blocks...)
    
    # Add the off-diagonal terms
    H1 = zeros(ComplexF64, 2*(2*Nm+1), 2*(2*Nm+1))
    for (i, n) in enumerate(n_range)
        for (j, m) in enumerate(m_range)
            diff = m - n
            block = (1.0im)^diff * besselj(diff, F) * cos(k - diff * π/2) * σ1   # fixed
            H1[2*i-1:2*i, 2*j-1:2*j] = block
        end
    end
    
    return H0 + H1
end

# Reference wavefunctions for the unperturbed system
function a0(k::Float64)
    return -1/√2 * √(1 - Q / √(Q^2 + cos(k)^2))
end

function b0(k::Float64)
    return 1/√2 * √(1 + Q / √(Q^2 + cos(k)^2))
end

function E0(k::Float64)
    return -√(Q^2 + cos(k)^2)
end

# Get eigenvector for a specific eigenvalue index (after sorting by real part)
function get_eigenvector(k::Float64, index::Int)
    eigen_decomp = eigen(H(k))
    sorted_indices = sortperm(eigen_decomp.values, by=x->real(x))
    return eigen_decomp.vectors[:, sorted_indices[index]]
end

function A(k::Float64)
    return get_eigenvector(k, 2*Nm + 1)
end

function B(k::Float64)
    return get_eigenvector(k, 2*Nm + 2)
end

# Projection sums
function Xa(k::Float64)
    vec = A(k)
    sum1 = sum(vec[1:2:end])
    sum2 = sum(vec[2:2:end])
    return [sum1, sum2]
end

function Xb(k::Float64)
    vec = B(k)
    sum1 = sum(vec[1:2:end])
    sum2 = sum(vec[2:2:end])
    return [sum1, sum2]
end

function wa(k::Float64)
    return conj(Xa(k)) ⋅ [a0(k), b0(k)]
end

function wb(k::Float64)
    return conj(Xb(k)) ⋅ [a0(k), b0(k)]
end

# Extract components from eigenvectors
function ChiA(k::Float64, j::Int)
    if -Nm - 1 < j < Nm + 1
        idx = 2*(j + Nm) + 1
        return [A(k)[idx], A(k)[idx+1]]
    else
        return [0.0+0.0im, 0.0+0.0im]
    end
end

function ChiB(k::Float64, j::Int)
    if -Nm - 1 < j < Nm + 1
        idx = 2*(j + Nm) + 1
        return [B(k)[idx], B(k)[idx+1]]
    else
        return [0.0+0.0im, 0.0+0.0im]
    end
end

# Current matrix elements
function IHknl(k::Float64, N0::Int, n::Int, l::Int)
    term1 = abs2(wa(k)) * (conj(ChiA(k, n - l + N0))' * σ1 * ChiA(k, n))
    term2 = abs2(wb(k)) * (conj(ChiB(k, n - l + N0))' * σ1 * ChiB(k, n))
    
    return (term1 + term2) * (1.0im)^l * besselj(l, F) * sin(k + l * π/2)   # fixed
end

function IHk(N0::Int, k::Float64)
    result = 0.0 + 0.0im
    for n in -Nm:Nm
        for l in -Nm:Nm
            result += IHknl(k, N0, n, l)
        end
    end
    return result
end

function IH(N0::Int)
    result = 0.0 + 0.0im
    for i in 0:Nk
        k = -π/2 + π/Nk * i
        result += IHk(N0, k)
    end
    return result
end

# Plotting functions (only work if Plots.jl is loaded)
function plot_bands(NN::Int)
    if !has_plots
        println("Plots not available. Install Plots.jl to use this function.")
        return
    end
    
    k_values = [i * π / (2 * NN) for i in 0:NN]
    band1 = Float64[]
    band2 = Float64[]
    
    for k in k_values
        eigen_decomp = eigen(H(k))
        sorted_vals = sort(real(eigen_decomp.values))
        push!(band1, sorted_vals[2*Nm + 1])
        push!(band2, sorted_vals[2*Nm + 2])
    end
    
    p = plot(k_values, [band1 band2], 
             label=["Band 1" "Band 2"],
             xlabel="k", ylabel="Energy",
             title="Band Structure")
    display(p)
    readline()
end

function plot_weight_difference(NN::Int)
    if !has_plots
        println("Plots not available. Install Plots.jl to use this function.")
        return
    end
    
    k_values = [i * π / (2 * NN) for i in 0:NN]
    differences = Float64[]
    
    for k in k_values
        diff = real(abs2(wb(k)) - abs2(wa(k)))
        push!(differences, diff)
    end
    
    p = plot(k_values, differences,
             label="Weight Difference",
             xlabel="k", ylabel="|wb|² - |wa|²",
             title="Weight Difference vs k")
    display(p)
end

# Example run (modify parameters as needed)
function run_example()
    global Nm, W, F, Q, Nk
    
    # Parameters from the notebook
    Nm = 3
    W = 1.0
    F = 5.0
    Q = 0.1
    
    println("Running with parameters:")
    println("Nm = $Nm, W = $W, F = $F, Q = $Q")
    
    # Try larger Nm and Nk for convergence
    Nm = 5
    Nk = 10
    N0_max=10
    println("\nNk = $Nk")
    print("Calculate current: ")
    I_N=zeros(Float64,N0_max)
    for N0 in ProgressBar(1:N0_max)
      I_N[N0]=abs(IH(N0))
    end
# Plot Current
    title("HHC spectrum")
    plot(I_N, label="I_H(N)")
    PyPlot.show()
    
    # Generate plots if Plots is available
    #if has_plots
    #    Nm = 3
    #    NN = 50
    #    plot_bands(NN)
    #    plot_weight_difference(NN)
    #end
end

# Uncomment to run
run_example()
