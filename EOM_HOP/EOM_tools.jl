#
# Density matrix EOM in the Wannier Gauge (TB approximation)
# Claudio Attaccalite (2023)
#
using LinearAlgebra
#
function polarization(rho,k_grid,lattice,P_gauge,TB_sol,Dip_h)
  s_dim=lattice.dim
  pol_t=zeros(Float64,s_dim)
  off_diag=.~I(TB_sol.h_dim)
  for ik in 1:k_grid.nk
    if P_gauge==W_gauge
       rho_in=WH_rotate(rho[:,:,ik],TB_sol.eigenvec[:,:,ik])
    else
      rho_in=rho[:,:,ik]
    end
    for id in 1:s_dim
       if P_gauge==W_gauge
          dip=WH_rotate(Dip_h[:,:,id,ik],TB_sol.eigenvec[:,:,ik]).*off_diag
        else
          dip=Dip_h[:,:,id,ik].*off_diag
        end
        pol_t[id]+=real.(sum(dip.*transpose(rho_in)))
    end
  end
  pol_t=pol_t/k_grid.nk
  return pol_t
end 


function current(rho,k_grid,lattice,P_gauge,TB_sol,H_h,A_h,∇H_w)
 #   
 s_dim=lattice.dim
 j_intra_t=zeros(Float64, s_dim, Threads.nthreads())
 j_inter_t=zeros(Float64, s_dim, Threads.nthreads())
 #
 if P_gauge==H_gauge
   Threads.@threads for ik in 1:k_grid.nk
     rho_t_H=transpose(rho[:,:,ik])
     t_id=Threads.threadid()
     for id in 1:s_dim
        ∇H_h=WH_rotate(∇H_w[:,:,id,ik],TB_sol.eigenvec[:,:,ik]) #.*off_diag
        j_intra_t[id,t_id]=j_intra_t[id,t_id]-real.(sum(∇H_h.*rho_t_H))
        commutator=A_h[:,:,id,ik]*H_h[:,:,ik]-H_h[:,:,ik]*A_h[:,:,id,ik]
        j_inter_t[id,t_id]=j_inter_t[id,t_id]-imag(sum(commutator.*rho_t_H))
     end
   end
 else
   if P_gauge==H_gauge
#  Current using the Hamiltonian gauge
     Threads.@threads for ik in 1:k_grid.nk
        rho_h=WH_rotate(rho[:,:,ik],TB_sol.eigenvec[:,:,ik])
        rho_h=transpose(rho_h)
        t_id=Threads.threadid()
        for id in 1:s_dim
           # 
           # Not sure if I should multiply for off_diag this is not clear
           # If I do not do so I get finite current after the pulse (that it is fine)
           # Probably introducing a LifeTime for the electron remove the necessity
           # of this off_diag or a windows for the current 
           #
           ∇H_h=WH_rotate(∇H_w[:,:,id,ik],TB_sol.eigenvec[:,:,ik]) #.*off_diag
           j_intra_t[id,t_id]=j_intra_t[id,t_id]-real.(sum(∇H_h.*rho_h))
           commutator=A_h[:,:,id,ik]*H_h[:,:,ik]-H_h[:,:,ik]*A_h[:,:,id,ik]
	   j_inter_t[id,t_id]=j_inter_t[id,t_id]-imag(sum(commutator.*rho_h))
	end
     end
   elseif P_gauge==W_gauge
#    Current using the Wannier gauge
     Threads.@threads for ik in 1:k_grid.nk
       t_id=Threads.threadid()
       rho_w=transpose(rho[:,:,ik])
       for id in 1:s_dim
          j_intra_t[id,t_id]=j_intra_t[id,t_id]-real.(sum(∇H_w[:,:,id,ik].*rho_w))
          commutator=A_w[:,:,id,ik]*TB_sol.H_w[:,:,ik]-TB_sol.H_w[:,:,ik]*A_w[:,:,id,ik]
	  j_inter_t[id,t_id]=j_inter_t[id,t_id]-imag(sum(commutator.*rho_w))
       end
     end
   end
  end
  #
  j_intra=zeros(Float64,s_dim)
  j_inter=zeros(Float64,s_dim)
  #
  for id in 1:s_dim
      j_intra[id]=sum(j_intra_t[id,:])/k_grid.nk
      j_inter[id]=sum(j_inter_t[id,:])/k_grid.nk
  end
  #
  j_intra_t=nothing
  j_inter_t=nothing
  return j_intra,j_inter
  #
end 

