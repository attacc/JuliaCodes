using LinearAlgebra
using SpecialFunctions  # For Bessel functions
using PyPlot
using ProgressBars


# Constants (assuming these are defined elsewhere)
Nm = 10  # Example value, replace with actual value
Q = 0.1  # Example value, replace with actual value
W = 0.25  # Example value, replace with actual value
F = 1.0  # Example value, replace with actual value
Nk = 10 # Example value, replace with actual value
N0_max=8

# Pauli matrices
σ0 = [1 0; 0 1]    # idensity
σ1 = [0 1; 1 0]    # sigma_x
σ3 = [1 0; 0 -1]   # sigma_z

# Define the Hamiltonian matrix H[k]
# Notice that this matrix is not Hermitian
# I found that if you force it to be Hermitian results are the same
# (see the commented lines)
#
function H_FLQ(k)
    H_matrix = zeros(ComplexF64, (2*(2*Nm+1), 2*(2*Nm+1)))
    for n in -Nm:Nm
       for m in -Nm:Nm
            if n == m
                H_matrix[2*(n+Nm)+1:2*(n+Nm)+2, 2*(m+Nm)+1:2*(m+Nm)+2] = (n * W * σ0 + Q * σ3)
            end
            H_matrix[2*(n+Nm)+1:2*(n+Nm)+2, 2*(m+Nm)+1:2*(m+Nm)+2] += (1.0im)^(m-n) * besselj(m-n, F) * cos(k - (m-n)*π/2) * σ1
        end
    end
    return H_matrix
end

function H_tb(k)
  H_tb = zeros(ComplexF64, (2, 2))
  H_tb = Q * σ3
  H_tb+= cos(k)*σ1
  return H_tb
end

# Define a0[k]
a0(k) = -1/sqrt(2) * sqrt(1 - Q / sqrt(Q^2 + cos(k)^2))

# Define b0[k]
b0(k) = 1/sqrt(2) * sqrt(1 + Q / sqrt(Q^2 + cos(k)^2))

# Define E0[k]
E0(k) = -sqrt(Q^2 + cos(k)^2)

# Define A[k] (eigenvector corresponding to the (2*Nm+1)-th eigenvalue)
function A(k)
     eig = eigen(H_FLQ(k))
     return eig.vectors[:, 2*Nm+1]
end

# Define Xa[k]
function Xa(k)
    A_vec = A(k)
    return [sum(A_vec[2i-1] for i in 1:2*Nm+1), sum(A_vec[2i] for i in 1:2*Nm+1)]
end

# Define wa[k]
kpts=range(0,pi/2.0,Nk)

wa(k) = conj(Xa(k)) ⋅ [a0(k), b0(k)]
function wa_diag(Xa_diag,TB_vec,k)
    wa_out=dot(Xa_diag, TB_vec[:,1])
    return wa_out
end

# Define B[k] (eigenvector corresponding to the (2*Nm+2)-th eigenvalue)
function B(k)
    eig = eigen(H_FLQ(k))
    return eig.vectors[:, 2*Nm+2]
end

# Define Xb[k]
function Xb(k)
    B_vec = B(k)
    return [sum(B_vec[2i-1] for i in 1:2*Nm+1), sum(B_vec[2i] for i in 1:2*Nm+1)]
end

function Xvec(eigenvecs)
    A_vec=eigenvecs[:,2*Nm+1]
    B_vec=eigenvecs[:,2*Nm+2]
    Xa_diag=[sum(A_vec[2i-1] for i in 1:2*Nm+1), sum(A_vec[2i] for i in 1:2*Nm+1)]
    Xb_diag=[sum(B_vec[2i-1] for i in 1:2*Nm+1), sum(B_vec[2i] for i in 1:2*Nm+1)]
    return Xa_diag,Xb_diag
end

# Define wb[k]
wb(k) = conj(Xb(k)) ⋅ [a0(k), b0(k)]
function wb_diag(Xb_diag,TB_vec,k)
    wb_out=dot(Xb_diag,TB_vec[:,1])
    return wb_out
end

FLQ_bands    =zeros(Float64,Nk, (2*(2*Nm+1)))
FLQ_vecs     =zeros(ComplexF64,Nk, (2*(2*Nm+1)),  (2*(2*Nm+1)) )
TB_bands    =zeros(Float64,Nk, 2)
TB_vecs     =zeros(ComplexF64,Nk, 2, 2)

print("Calculate band structure: ")
for ik in ProgressBar(1:length(kpts))
#    print("Doing $k is H_FLQ(k) hermitian $(ishermitian(H(k))) \n")
    diag_FLQ = eigen(H_FLQ(kpts[ik]))
    FLQ_bands[ik,:]  =real(diag_FLQ.values)
    FLQ_vecs[ik,:,:] =diag_FLQ.vectors
    diag_TB = eigen(H_tb(kpts[ik]))
    TB_bands[ik,:]  =real(diag_TB.values)
    TB_vecs[ik,:,:] =diag_TB.vectors
end

title("Tight binding band structure ")
for n in 1:2
  plot(kpts,TB_bands[:,n], label="Band $n")
end
PyPlot.show()


title("Floquet band structure ")
for n in 1:2*(2*Nm+1)
  plot(kpts,FLQ_bands[:,n], label="Band $n")
end
PyPlot.show()

print("Calculate Wa and Wb: ")
wa_d=zeros(ComplexF64,Nk)
wb_d=zeros(ComplexF64,Nk)
wa2=zeros(Float64,Nk)
wb2=zeros(Float64,Nk)
for ik in ProgressBar(1:length(kpts))
    Xa_diag,Xb_diag=Xvec(FLQ_vecs[ik,:,:])
    wa_d[ik]=wa_diag(Xa_diag,TB_vecs[ik,:,:],kpts[ik])
    wb_d[ik]=wb_diag(Xb_diag,TB_vecs[ik,:,:],kpts[ik])    
    wa2[ik]=abs(wa_d[ik]^2)
    wb2[ik]=abs(wb_d[ik]^2)    
end

plot(kpts,wb2-wa2, label="W difference")
plot(kpts,wb2+wa2, label="W sum")
PyPlot.show()



# Define Bound[l]
Bound(l) = (-Nm-1 < l < Nm+1) ? 1 : 0

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

# IHknl now receives eigenvectors and wa/wb values
function IHknl(k::Float64, N0::Int, n::Int, l::Int)
    term1 = abs2(wa(k)) * (conj(ChiA(k, n-l+N0)) ⋅ (σ1 * ChiA(k, n)))
    term2 = abs2(wb(k)) * (conj(ChiB(k, n-l+N0)) ⋅ (σ1 * ChiB(k, n)))
    return (term1 + term2) * (1.0im)^l * besselj(l, F) * sin(k + l*π/2)
end

# IHk computes eigenvectors once per k
function IHk(N0::Int, k::Float64)
    total = 0.0+0.0im
    for n in -Nm:Nm, l in -Nm:Nm
        total += IHknl(k, N0, n, l)
    end
    return total
end

# IH sums over k points (ensure correct range and use pi, not π)
function IH(N0::Int)
    dk = pi / Nk   # spacing, but note the original used -π/2 + π/Nk * i for i=0..Nk
    total = 0.0+0.0im
    for i in 0:Nk
        k = -pi/2 + dk * i
        total += IHk(N0, k)
    end
    return total
end


# Calculate Current
print("Calculate current: ")
I_N=zeros(Float64,N0_max)
for N0 in ProgressBar(1:N0_max)
    I_N[N0]=abs(IH(N0))
end

# Plot Current
title("HHC spectrum")
plot(I_N, label="I_H(N)")
PyPlot.show()

