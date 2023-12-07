#
# Density matrix EOM in the Wannier Gauge (TB approximation)
# Claudio Attaccalite (2023)
#
using LinearAlgebra
using CSV
using DataFrames
using Base.Threads
using PyPlot


include("TB_hBN.jl")
using .hBN2D

include("TB_tools.jl")
using .TB_tools

include("units.jl")
using .Units

include("lattice.jl")
using .LatticeTools

lattice=set_Lattice(2,[a_1,a_2])
# 
# Code This code is in Hamiltonian space
# in dipole approximation only at the K-point
#
# * * * DIPOLES * * * #
#
# if use_Dipoles=true  dipoles are calculated
# using dH/dh
#
# if use_Dipoles=false dipoles are calculated
# uding UdU with fixed phase
#
use_Dipoles=true


# a generic off-diagonal matrix example (0 1; 1 0)
off_diag=.~I(h_dim)

# K-points
n_k1=96
n_k2=96

k_grid=generate_unif_grid(n_k1, n_k2, lattice)

nk=n_k1*n_k2
#nk=1
#k_vec=[1,1/sqrt(3)]
#k_vec=k_vec*2*pi/(3*a_cc)
#k_list=k_vec

eigenval=zeros(Float64,s_dim,nk)
eigenvec=zeros(Complex{Float64},s_dim,s_dim,nk)

println(" K-point list ")
println(" nk = ",nk)

println("Building Hamiltonian: ")
Threads.@threads for ik in ProgressBar(1:nk)
        H_w=Hamiltonian(k_grid.kpt[:,ik])
	data= eigen(H_w)      # Diagonalize the matrix
	eigenval[:,ik]   = data.values
	eigenvec[:,:,ik] = data.vectors
	eigenvec[:,:,ik] = fix_eigenvec_phase(eigenvec[:,:,ik])
end

#Hamiltonian info
dir_gap=minimum(eigenval[2,:]-eigenval[1,:])
println("Direct gap : ",dir_gap*ha2ev," [eV] ")
ind_gap=minimum(eigenval[2,:])-maximum(eigenval[1,:])
println("Indirect gap : ",ind_gap*ha2ev," [eV] ")

# rotate in the Hamiltonian guage
Dip_h=zeros(Complex{Float64},h_dim,h_dim,nk,s_dim)

if use_Dipoles
  println("Building Dipoles using dH/dk:")
  Threads.@threads for ik in ProgressBar(1:nk)
# Dipoles
    Dip_w=Grad_H(k_grid.kpt[:,ik])
    for id in 1:s_dim
          Dip_h[:,:,ik,id]=HW_rotate(Dip_w[:,:,id],eigenvec[:,:,ik],"W_to_H")
# I set to zero the diagonal part of dipoles
          Dip_h[:,:,ik,id]=Dip_h[:,:,ik,id].*off_diag
    end

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
            Dip_h[i,j,ik,:]= 1im*Dip_h[i,j,ik,:]/(eigenval[j,ik]-eigenval[i,ik])
            Dip_h[j,i,ik,:]=conj(Dip_h[i,j,ik,:])
  	end
     end
  end
else
  println("Building Dipoles using UdU/dk:")
  Threads.@threads for ik in ProgressBar(1:nk)
     UdU=k_deriv_to_cart(Calculate_UdU(k_list[:,ik], eigenvec[:,:,ik]), lattice)
     for id in 1:s_dim
        Dip_h[:,:,ik,id]=UdU[:,:,id]
     end
  end
end

E_field_ver=[1.0,0.0]
freqs_range  =[0.0/ha2ev, 15.0/ha2ev] # eV
eta          =0.15/ha2ev
freqs_nsteps =200
freqs=LinRange(freqs_range[1],freqs_range[2],freqs_nsteps)

function Linear_response(freqs,E_field_ver, eta)
   nv=1
   xhi=zeros(Complex{Float64},length(freqs))
   Res=zeros(Complex{Float64},h_dim,h_dim,nk)

   println("Residuals: ")
   Threads.@threads for ik in ProgressBar(1:nk)
   for iv in 1:nv,ic in nv+1:h_dim
      Res[iv,ic,ik]=sum(Dip_h[iv,ic,ik,:].*E_field_ver[:])
   end
   end
   print("Xhi: ")
   Threads.@threads for ifreq in ProgressBar(1:length(freqs))
     for ik in 1:nk,iv in 1:nv,ic in nv+1:h_dim
         xhi[ifreq]=xhi[ifreq]+abs(Res[iv,ic,ik])^2/(eigenval[ic,ik]-eigenval[iv,ik]-freqs[ifreq]-eta*1im)
     end
   end
   xhi.=xhi/nk
   return xhi
end

xhi = Linear_response(freqs, E_field_ver, eta)

fig = figure("Linear response",figsize=(10,20))
plot(freqs*ha2ev,real(xhi[:]))
plot(freqs*ha2ev,imag(xhi[:]))
PyPlot.show();

df = DataFrame(freq  = freqs*ha2ev,
               im_xhi = imag(xhi[:]),
               re_rhi = real(xhi[:]),
               )
f = open("xhi_w.csv","w")
CSV.write(f, df; quotechar=' ', delim=' ')
