#
# Dipole matrix elements (Berry connection)
# Claudio Attaccalite (2023)
#
using LinearAlgebra
using Base.Threads
#
#
# * * * DIPOLES * * * #
#
# if use_gradH=true  dipoles are calculated
# using dH/dh
#
# if use_GradH=false dipoles are calculated
# uding UdU with fixed phase
#
#
function Build_Dipole(k_grid,lattice,TB_sol,TB_gauge,orbitals,Hamiltonian,dk,use_GradH)
  #      
  if use_GradH
    println("Building Dipoles using dH/dk:")
  else
    println("Building Dipoles using UdU/dk:")
  end
  # 
  # a generic off-diagonal matrix example (0 1; 1 0)
  #
  s_sim=lattice.dim
  h_dim=TB_sol.h_dim
  off_diag=.~I(h_dim)
  #
  Dip_h=zeros(Complex{Float64},h_dim,h_dim,s_dim,k_grid.nk)
  ∇H_w =zeros(Complex{Float64},h_dim,h_dim,s_dim,k_grid.nk)
  Threads.@threads for ik in ProgressBar(1:k_grid.nk)
    #  
    if use_GradH
      #  
      ∇H_w[:,:,:,ik]=Grad_H(ik,k_grid,lattice,TB_sol,TB_gauge; Hamiltonian=Hamiltonian,deltaK=dk)
      #
      for id in 1:s_dim
        Dip_h[:,:,id,ik]=WH_rotate(∇H_w[:,:,id,ik],TB_sol.eigenvec[:,:,ik])
# I set to zero the diagonal part of dipoles
        Dip_h[:,:,id,ik]=Dip_h[:,:,id,ik].*off_diag
      end
      #
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
            Dip_h[i,j,:,ik]= 1im*Dip_h[i,j,:,ik]/(TB_sol.eigenval[j,ik]-TB_sol.eigenval[i,ik])
            Dip_h[j,i,:,ik]=conj(Dip_h[i,j,:,ik])
  	end
      end
      #
    else
      #
      #  Build dipoles using U\grad U
      #
      ∇U  =Grad_U(ik,k_grid,lattice,TB_sol,TB_gauge; Hamiltonian=Hamiltonian,deltaK=dk)
      #
      U=TB_sol.eigenvec[:,:,ik]
      for id in 1:s_dim
        Dip_h[:,:,id,ik]=1im*(U')*∇U[:,:,id]
      end
      # 
    end
    #
    # Gauge correction to the \grad U in the lattice gauge
    #
    if TB_gauge==TB_lattice
        VdV=Gauge_Correction(TB_sol,orbitals)
        do id in 1:s_dim
            Dip_h[:,:,id,ik]=1im*(U')*VdV[:,:,id]
        end
    end
    #
  end
  #
  return Dip_h,∇H_w
  #
end
