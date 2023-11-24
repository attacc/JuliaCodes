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
# 
# Code This code is in Hamiltonian space
# in dipole approximation only at the K-point
#
# Hamiltonian dimension
h_dim=2
# Space dimension
s_dim=2



# a generic off-diagonal matrix example (0 1; 1 0)
off_diag=.~I(h_dim)

# K-points
n_k1=96
n_k2=96

a_1=[    sqrt(3.0)     , 0.0]
a_2=[ sqrt(3.0)/2.0, 3.0/2.0]

b_1=2*pi/3.0*[ sqrt(3.0),  -1.0 ]
b_2=2*pi/3.0*[ 0.0,         2.0 ]

#
# Matrix to pass from crystal to cartesian
#
b_mat=zeros(Float64,s_dim,s_dim)
b_mat[:,1]=b_1
b_mat[:,2]=b_2


k_list=generate_unif_grid(n_k1, n_k2, b_mat)

nk=n_k1*n_k2

eigenval=zeros(Float64,s_dim,nk)
eigenvec=zeros(Complex{Float64},s_dim,s_dim,nk)

println(" K-point list ")
println(" nk = ",nk)
# for ik in 1:nk
# 	println(k_list[:,ik])
# end

#K=[1.0/3.0,-1.0/3.0]
#K_cc=b_mat*K
#K_cc=1.0/3.0*b_1+2.0/3.0*b_2
#println(K_cc)
#K_t=2*pi/3.0* [1/sqrt(3.0), 1]
#println(K_t)
#K_t2=2*pi/3.0*[-1/sqrt(3.0), 1]
#println(K_t2)

#h_k=Hamiltonian(K_cc)
#println(h_k[1,1]*ha2ev,"---",h_k[1,2]*ha2ev)
#println(h_k[2,1]*ha2ev,"---",h_k[2,2]*ha2ev)
#	data= eigen(h_k)      # Diagonalize the matrix
#        println(data.values*ha2ev)
#h_k=Hamiltonian(K_t2)
#println(h_k[1,1]*ha2ev,"---",h_k[1,2]*ha2ev)
#println(h_k[2,1]*ha2ev,"---",h_k[2,2]*ha2ev)
#	data= eigen(h_k)      # Diagonalize the matrix
#        println(data.values*ha2ev)
#exit()

H_h=zeros(Complex{Float64},h_dim,h_dim,nk)
println("Building Hamiltonian: ")
Threads.@threads for ik in ProgressBar(1:nk)
        H_w=Hamiltonian(k_list[:,ik])
	data= eigen(H_w)      # Diagonalize the matrix
	eigenval[:,ik]   = data.values
	eigenvec[:,:,ik] = data.vectors
        for i in 1:h_dim
           H_h[i,i,ik]=eigenval[i,ik]
        end
end

# rotate in the Hamiltonian guage
Dip_h=zeros(Complex{Float64},h_dim,h_dim,nk,s_dim)

println("Building Dipoles: ")
Threads.@threads for ik in ProgressBar(1:nk)
# Dipoles
  Dip_w=Grad_H(k_list[:,ik])
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



E_field_ver=[1.0,1.0]
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
   return xhi
end

xhi = Linear_response(freqs, E_field_ver, eta)

fig = figure("Linear response",figsize=(10,20))
plot(freqs*ha2ev,real(xhi[:]))
plot(freqs*ha2ev,imag(xhi[:]))
PyPlot.show();


