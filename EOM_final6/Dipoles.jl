#
# Dipole matrix elements (Berry connection)
# Claudio Attaccalite (2023)
#
using LinearAlgebra
using Base.Threads

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
  # a generic off-diagonal matrix example (0 1; 1 0)
  off_diag=.~I(h_dim)
  #
  Dip_h=zeros(Complex{Float64},h_dim,h_dim,k_grid.nk,s_dim)
  Threads.@threads for ik in ProgressBar(1:k_grid.nk)
    #  
    if(use_GradH) 
      ∇H_w=Grad_H(ik,k_grid,lattice,TB_sol,TB_gauge; orbitals=orbitals, Hamiltonian=Hamiltonian,deltaK=dk)
    else
      ∇U  =Grad_U(ik,k_grid,lattice,TB_sol,TB_gauge; orbitals=orbitals, Hamiltonian=Hamiltonian,deltaK=dk)
    end
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
  return Dip_h
  #
end
