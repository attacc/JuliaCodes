#
# Density matrix EOM in the Wannier Gauge (TB approximation)
# Claudio Attaccalite (2023)
#
using LinearAlgebra
using DataFrames
using Base.Threads
using PyPlot

include("EOM_input.jl")
#
include("Dipoles.jl")
#
include("TB_tools.jl")
# 
# Generate the Monkhorst-Pack grid
#
k_grid=generate_unif_grid(n_k1, n_k2, lattice)
#
# Solve the Hamiltonian on the grid and store the solution in TB_sol
#
TB_sol=Solve_TB_on_grid(k_grid,BN_Hamiltonian,TB_gauge)
#
# Calculate Dipoles
#
Dip_h,∇H_w=Build_Dipole(k_grid,lattice,TB_sol,TB_gauge,BN_orbitals,BN_Hamiltonian,dk,use_GradH)
#
freqs=LinRange(freqs_range[1],freqs_range[2],freqs_nsteps)
#
# Function that calculate the linear respone
#
function Linear_response(freqs,E_field_ver, η)
   nv=1
   xhi=zeros(Complex{Float64},length(freqs))
   Res=zeros(Complex{Float64},h_dim,h_dim,k_grid.nk)

   println("Residuals: ")
   Threads.@threads for ik in ProgressBar(1:k_grid.nk)
     for iv in 1:nv,ic in nv+1:h_dim
        Res[iv,ic,ik]=sum(Dip_h[iv,ic,:,ik].*E_field_ver[:])
     end
   end
   print("Xhi: ")
   Threads.@threads for ifreq in ProgressBar(1:length(freqs))
     for ik in 1:k_grid.nk,iv in 1:nv,ic in nv+1:h_dim
         e_v=TB_sol.eigenval[iv,ik]
         e_c=TB_sol.eigenval[ic,ik]
         xhi[ifreq]=xhi[ifreq]+abs(Res[iv,ic,ik])^2/(e_c-e_v-freqs[ifreq]-η*1im)
     end
   end
   xhi.=xhi/k_grid.nk
   return xhi
end

function generate_header(k_grid,eta,Efield_ver,freqs)
    header="#\n# * * * Linear response xhi(ω) * * * \n#\n" 
    header*="# k-point grid: $(k_grid.nk_dir[1]) - $(k_grid.nk_dir[2]) \n"
    header*="# E-field versor: $(Efield_ver) \n"
    header*="# Frequencies range: $(freqs[1]*ha2ev) - $(freqs[end]*ha2ev) [eV]  \n"
    header*="# Frequencies steps: $(length(freqs)) \n#\n"
    return header
end

xhi = Linear_response(freqs, e_field.E_vec, eta)
# 
# Plot and write on disk
#
fig = figure("Linear response",figsize=(10,20))
plot(freqs*ha2ev,real(xhi[:]))
plot(freqs*ha2ev,imag(xhi[:]))
PyPlot.show();


f = open("xhi_w.csv","w")
header=generate_header(k_grid,eta,e_field.E_vec,freqs)
write(f,header)
for iw in 1:freqs_nsteps
    write(f," $(freqs[iw]*ha2ev) $(imag(xhi[iw])) $(real(xhi[iw])) \n")
end         
close(f)
