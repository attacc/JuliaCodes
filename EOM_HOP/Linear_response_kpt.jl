#
# Density matrix EOM in the Wannier Gauge (TB approximation)
# Claudio Attaccalite (2023)
#
using LinearAlgebra
using DataFrames
using Base.Threads
using PyPlot

include("EOM_input.jl")

lattice =set_Lattice(2,[a_1,a_2])
# 
# Code This code is in Hamiltonian space
# in dipole approximation only at the K-point
#
# * * * DIPOLES * * * #
#
# if use_gradH=true  dipoles are calculated
# using dH/dh
#
# if use_GradH=false dipoles are calculated
# uding UdU with fixed phase
#
use_GradH=false

# a generic off-diagonal matrix example (0 1; 1 0)
off_diag=.~I(h_dim)

k_grid=generate_unif_grid(n_k1, n_k2, lattice)

#nk=1
#k_vec=[1,1/sqrt(3)]
#k_vec=k_vec*2*pi/(3*a_cc)
#k_list=k_vec
TB_sol.h_dim=2
TB_sol.eigenval=zeros(Float64,h_dim,k_grid.nk)
TB_sol.eigenvec=zeros(Complex{Float64},h_dim,h_dim,k_grid.nk)
TB_sol.H_w     =zeros(Complex{Float64},h_dim,h_dim,k_grid.nk)

println(" K-point list ")
println(" nk = ",k_grid.nk)
#print_k_grid(k_grid, lattice)
#
println("Tight-binding gauge : $TB_gauge ")
println("Delta-k for derivatives : $dk ")

println("Building Hamiltonian: ")
Threads.@threads for ik in ProgressBar(1:k_grid.nk)
   TB_sol.H_w[:,:,ik]=Hamiltonian(k_grid.kpt[:,ik],TB_gauge)
   data= eigen(TB_sol.H_w[:,:,ik])      # Diagonalize the matrix
   TB_sol.eigenval[:,ik]   = data.values
   TB_sol.eigenvec[:,:,ik] = data.vectors
   TB_sol.eigenvec[:,:,ik] = fix_eigenvec_phase(TB_sol.eigenvec[:,:,ik])
end
#
#Print Hamiltonian info
#
dir_gap=minimum(TB_sol.eigenval[2,:]-TB_sol.eigenval[1,:])
println("Direct gap : ",dir_gap*ha2ev," [eV] ")
ind_gap=minimum(TB_sol.eigenval[2,:])-maximum(TB_sol.eigenval[1,:])
println("Indirect gap : ",ind_gap*ha2ev," [eV] ")

if use_GradH
  println("Building Dipoles using dH/dk:")
else
  println("Building Dipoles using UdU/dk:")
end
# 
# Dipoles
#
Dip_h=zeros(Complex{Float64},h_dim,h_dim,k_grid.nk,s_dim)
Threads.@threads for ik in ProgressBar(1:k_grid.nk)
  #  
  ∇H_w=Grad_H(ik,k_grid,lattice,Hamiltonian,TB_sol,TB_gauge,dk)
  #
  ∇U=Grad_U(ik,k_grid,lattice,TB_sol,TB_gauge,dk)
  #
  if use_GradH
    for id in 1:s_dim
      Dip_h[:,:,ik,id]=WH_rotate(∇H_w[:,:,id],TB_sol.eigenvec[:,:,ik])
# I set to zero the diagonal part of dipoles
      Dip_h[:,:,ik,id]=Dip_h[:,:,ik,id].*off_diag
    end
#    
# Now I have to divide for the energies
#
#  p = \grad_k H 
#
#  r_{ij} = i * p_{ij}/(e_j - e_i)
#
#  (diagonal terms are set to zero)
#
    for i in 1:h_dim
      for j in i+1:h_dim
            Dip_h[i,j,ik,:]= 1im*Dip_h[i,j,ik,:]/(TB_sol.eigenval[j,ik]-TB_sol.eigenval[i,ik])
            Dip_h[j,i,ik,:]=conj(Dip_h[i,j,ik,:])
  	end
    end
  else
     #
     #  Build dipoles using U\grad U
     #
     U=TB_sol.eigenvec[:,:,ik]
     for id in 1:s_dim
        Dip_h[:,:,ik,id]=1im*(U')*∇U[:,:,id]
     end
  end
  #
end
#
freqs=LinRange(freqs_range[1],freqs_range[2],freqs_nsteps)
#
# Function that calculate the linear respone
#
function Linear_response(freqs,E_field_ver, eta)
   nv=1
   xhi=zeros(Complex{Float64},length(freqs))
   Res=zeros(Complex{Float64},h_dim,h_dim,k_grid.nk)

   println("Residuals: ")
   Threads.@threads for ik in ProgressBar(1:k_grid.nk)
     for iv in 1:nv,ic in nv+1:h_dim
        Res[iv,ic,ik]=sum(Dip_h[iv,ic,ik,:].*E_field_ver[:])
     end
   end
   print("Xhi: ")
   Threads.@threads for ifreq in ProgressBar(1:length(freqs))
     for ik in 1:k_grid.nk,iv in 1:nv,ic in nv+1:h_dim
         e_v=TB_sol.eigenval[iv,ik]
         e_c=TB_sol.eigenval[ic,ik]
         xhi[ifreq]=xhi[ifreq]+abs(Res[iv,ic,ik])^2/(e_c-e_v-freqs[ifreq]-eta*1im)
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

xhi = Linear_response(freqs, E_vec, eta)
# 
# Plot and write on disk
#
fig = figure("Linear response",figsize=(10,20))
plot(freqs*ha2ev,real(xhi[:]))
plot(freqs*ha2ev,imag(xhi[:]))
PyPlot.show();


f = open("xhi_w.csv","w")
header=generate_header(k_grid,eta,E_vec,freqs)
write(f,header)
for iw in 1:freqs_nsteps
    write(f," $(freqs[iw]*ha2ev) $(imag(xhi[iw])) $(real(xhi[iw])) \n")
end         
close(f)
