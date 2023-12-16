module TB_tools

using Printf
using ProgressBars

export evaluate_DOS,rungekutta2_dm,fix_eigenvec_phase,ProgressBar,K_crys_to_cart,props,IO_output,dyn_props,TB_sol,Grad_H_and_U

mutable struct K_Grid
	kpt::Array{Float64,2}
	nk_dir::Array{Int,1}
	ik_map::Array{Int,3}
	ik_map_inv::Array{Int,2}
end

mutable struct Properties
	eval_curr   ::Bool
	eval_pol    ::Bool
	print_dm    ::Bool
end

mutable struct Dynamics_Properties
	h_gauge      :: Bool
	damping	     :: Bool
	use_dipoles  :: Bool
	include_drho_dk :: Bool
	include_A_w	:: Bool
end

mutable struct TB_Solution
        h_dim::Int
	eigenval::Array{Float64,2}
	eigenvec::Array{Complex{Float64},3}
	H_w::Array{Complex{Float64},3}
	TB_Solution() = new()
end

TB_sol=TB_Solution()

dyn_props = Dynamics_Properties(true,true,false,true,true)

props = Properties(true, true, false)

mutable struct IO_Output
	dm_file  ::Union{IOStream, Nothing}
	IO_Output(dm_file=nothing) =new(dm_file)
end

IO_output=IO_Output()

function generate_circuit(points, n_steps)
	println("Generate k-path ")
	n_points=length(points)
	@printf("number of points:%5i \n",n_points)
	if n_points <= 1
		error("number of points less or equal to 1 ")
	end
	for i in 1:n_points
		@printf("v(%d) = %s \n",i,points[i])
	end
	path=Any[]
	for i in 1:(n_points-1)
		for j in 0:(n_steps-1)	
			dp=(points[i+1]-points[i])/n_steps
			push!(path,points[i]+dp*j)
		end
	end
	push!(path,points[n_points])
	return path
 end
 #
 #
 function fermi_function(E, E_f, Temp)
   fermi_function=1.0/((exp(E-E_f)/Temp))
   return fermi_function
 end
 #
 function generate_unif_grid(n_kx, n_ky, lattice)
     nk   =n_kx*n_ky
     kpt  =zeros(Float64,lattice.dim,nk)
     ik_grid    =zeros(Int,n_kx,n_ky,1)
     ik_grid_inv=zeros(Int,lattice.dim,nk)

     vec_crystal=zeros(Float64,lattice.dim)
     dx=1.0/n_kx
     dy=1.0/n_ky

     ik=1
     for ix in 0:(n_kx-1),iy in 0:(n_ky-1)
           ik_grid[ix+1,iy+1,1]=ik
           ik_grid_inv[:,ik]   =[ix+1,iy+1]
	   vec_crystal[1]=dx*ix
	   vec_crystal[2]=dy*iy
	   kpt[:,ik]=lattice.rvectors[1]*vec_crystal[1]+lattice.rvectors[2]*vec_crystal[2]
	   ik=ik+1
     end

     k_grid = K_Grid(
		kpt,
		[n_kx,n_ky,1],
		ik_grid,
		ik_grid_inv
	       )
     return k_grid
 end
 #
 function print_k_grid(k_grid, lattice)
    println("K-points grid ")
    println("grid dimensions : ",k_grid.nk_dir, "\n\n")
    for ik in 1:size(k_grid.kpt, 2)
       k_crystal=lattice.b_mat_inv*k_grid.kpt[:,ik]
       println("ik ",ik," kpt ",k_crystal," ik_grid ",k_grid.ik_map_inv[:,ik])
       ik_left =get_k_neighbor(ik,1,1,k_grid)
       ik_right=get_k_neighbor(ik,1,-1,k_grid)
       println("x-neighboar ",ik_left," - ",ik_right)
    end
 end
 #
 function get_k_neighbor(ik,id,istep,k_grid)
     s_dim=2

     ik_xyz   =k_grid.ik_map_inv[:,ik]
     ik_n     =ik_xyz
     ik_n[id] =ik_n[id]+istep 
#
     d_size=k_grid.nk_dir[id]
     if ik_n[id] > d_size
        ik_n[id]=ik_n[id]-d_size
     elseif ik_n[id]<=0
        ik_n[id]=ik_n[id]+d_size
     end
#
     ik_indx=[1,1,1]
     for id in 1:s_dim
        ik_indx[id]=ik_n[id]
     end
     ik_n=k_grid.ik_map[ik_indx[1],ik_indx[2],ik_indx[3]]
     return ik_n
 end
 #
 function lorentzian(x,x_0, Gamma)
	 return 1.0/pi*(0.5*Gamma)/((x-x_0)^2+(0.5*Gamma)^2)
 end
 #
 function gaussian(x,x_0, Sigma)
	 return 1.0/(Sigma*sqrt(2.0*pi))*exp(-(x-x_0)^2/(2*Sigma^2))
 end
 #
 function evaluate_DOS(bands::Matrix{Float64},E_range::Vector{Float64}, n_points::Integer, smearing::Float64)
	dE=(E_range[2]-E_range[1])/n_points
        DOS=zeros(Float64,n_points,2)
	for i in 1:n_points
		e_dos=E_range[1]+dE*(i-1)
		DOS[i,1]=e_dos
		for eb in bands
#			DOS[i,2]=DOS[i,2]+lorentzian(e_dos,eb,smearing)
			DOS[i,2]=DOS[i,2]+gaussian(e_dos,eb,smearing)
		end
	end
	return DOS
 end
 #
end

function HW_rotate(M,eigenvec,mode)
	if mode=="W_to_H"
		rot_M=(eigenvec\M)*eigenvec
	elseif mode=="H_to_W"
		rot_M=(eigenvec*M)/eigenvec
	else
		println("Wrong call to rotate_H_to_W")
		exit()
	end
	return rot_M
end 

function init_output()
	if props.print_dm    
      		println("Write density matrix on disk for k=1")
		IO_output.dm_file  =open("density_matrix.txt","w")	
	end
end

function print_density_matrix(time,rho_i)
        ik=1
	if dyn_props.h_gauge
		rho=rho_i[:,:,2]
	else
		rho=HW_rotate(rho_i[:,:,2],TB_sol.eigenvec[:,:,2],"W_to_H")
	end
	write(IO_output.dm_file," $(time) ")
	write(IO_output.dm_file," $(real(rho[1,1])) $(imag(rho[1,1])) ")
	write(IO_output.dm_file," $(real(rho[1,2])) $(imag(rho[1,2])) ")
	write(IO_output.dm_file," $(real(rho[2,1])) $(imag(rho[2,1])) ")
	write(IO_output.dm_file," $(real(rho[2,2])) $(imag(rho[2,2])) \n")
end

function finalize_output()
	if IO_output.dm_file != nothing  
		close(IO_output.dm_file)	
	end
end 

function rungekutta2_dm(d_rho, rho_0, t)
    n     = length(t)
    nk    = size(rho_0)[3]
    h_dim = size(rho_0)[1]
    rho_t = zeros(Complex{Float64},n, h_dim, h_dim, nk)
    rho_t[1,:,:,:] = rho_0

    init_output()

    println("Real-time equation integration: ")
    for i in ProgressBar(1:n-1)
        h = t[i+1] - t[i]
	rho_t[i+1,:,:,:] = rho_t[i,:,:,:] + h * d_rho(rho_t[i,:,:,:] + d_rho(rho_t[i,:,:,:], t[i]) * h/2, t[i] + h/2)

	if props.print_dm
	   print_density_matrix(t[i],rho_t[i,:,:,:])
	end
    end

    finalize_output()

    return rho_t
end
#
# Fix phase of the eigenvectors in such a way
# to have a positive definite diagonal
#
function fix_eigenvec_phase(eigenvec)
#	println("Before phase fixing : ")
#	show(stdout,"text/plain",eigenvec)
	#
	# Rotation phase matrix
	#
	phase_m=zeros(Complex{Float64},2,2)
	phase_m[1,1]=exp(-1im*angle(eigenvec[1,1]))
        phase_m[2,2]=exp(-1im*angle(eigenvec[2,2]))
	#
	# New eigenvectors
	#
	eigenvec=eigenvec*phase_m
#	println("\nAfter phase fixed : ")
#	show(stdout,"text/plain",eigenvec)
	#
	# Norm
	#
#	println(norm(eigenvec[:,1]))
#	println(norm(eigenvec[:,2]))
	#
	return eigenvec
end

# Based on perturbation theory
# Eq. 24 of https://arxiv.org/pdf/cond-mat/0608257.pdf
#
function Grad_H_and_U(ik, k_grid, lattice, TB_sol, dk=0.01, Hamiltonian=nothing)
    #
    # calculate dH/dk in the Wannier Gauge
    # derivatives are in cartesian coordinates
    #
    h_dim=TB_sol.h_dim    # hamiltonian dimension
    s_dim=lattice.dim     # space dimension
    #
    dH_w       =zeros(Complex{Float64},h_dim,h_dim,s_dim)
    dH_eigenval=zeros(Float64,h_dim,s_dim)
    dU         =zeros(Complex{Float64},h_dim,h_dim,s_dim)
    #
    if dk !=0 && Hamiltonian != nothing
      #    
      # Derivative by finite differences, 
      # recalculating the Hamiltonian
      #
      k_plus =copy(k_grid.kpt[:,ik])
      k_minus=copy(k_grid.kpt[:,ik])
      #
      for id in 1:s_dim
        #  
#       k_plus[id] =k_grid.kpt[id,ik]+dk
#       k_minus[id]=k_grid.kpt[id,ik]-dk
#       #
        if k_grid.nk_dir[id]==1
           continue
        end
#       # 
        k_plus +=lattice.rvectors[id]/lattice.rv_norm[id]*dk
        k_minus-=lattice.rvectors[id]/lattice.rv_norm[id]*dk
        #
        H_plus =Hamiltonian(k_plus)
	data_plus= eigen(H_plus)
        eigenval_p = data_plus.values
        eigenvec_p = data_plus.vectors
	eigenvec_p= fix_eigenvec_phase(eigenvec_p)
        #
        H_minus=Hamiltonian(k_minus)
	data_minus= eigen(H_minus)
        eigenval_m = data_minus.values
	eigenvec_m = data_minus.vectors
	eigenvec_m= fix_eigenvec_phase(eigenvec_m)
        #
        k_plus =copy(k_grid.kpt[:,ik])
        k_minus=copy(k_grid.kpt[:,ik])
        #     
        dH_w[:,:,id]=(H_plus-H_minus)/(2.0*dk)
        #
        dU[:,:,id]=(eigenvec_p-eigenvec_m)/(2.0*dk)
        #
        dH_eigenval[:,id]=(eigenval_p-eigenval_m)/(2.0*dk)
      end
      #
      # Convert from crystal to cartesian
      #
      dH_w       =K_crys_to_cart(dH_w,lattice)
      dU         =K_crys_to_cart(dU,lattice)
      dH_eigenval=K_crys_to_cart(dH_eigenval,lattice)
      #
    else
      #
      # Derivative by finite differences, 
      # using the regular k-grid 
      #
      for id in 1:s_dim
          #
          if k_grid.nk_dir[id]==1
               continue
          end
          #
          # Derivatives using the k-grid in input
          # along the reciprocal lattice vectors
          # dk is the distance between the nearest 
          # k-points dk=k[i]-k[j]
          #
          ik_plus =get_k_neighbor(ik,id, 1,k_grid)
          eigenvec_p = TB_sol.eigenvec[:,:,ik_plus]
          eigenval_p = TB_sol.eigenval[:,ik_plus]
          H_plus  =TB_sol.H_w[:,:,ik_plus]
          #
          ik_minus=get_k_neighbor(ik,id,-1,k_grid)
          eigenvec_m = TB_sol.eigenvec[:,:,ik_minus]
          eigenval_m = TB_sol.eigenval[:,ik_minus]
          H_minus =TB_sol.H_w[:,:,ik_minus]
          #
          dk=lattice.rv_norm[id]/k_grid.nk_dir[id]/2.0
          #
          dH_w[:,:,id]=(H_plus-H_minus)/(2.0*dk)
          #
          dU[:,:,id]=(eigenvec_p-eigenvec_m)/(2.0*dk)
          #
          dH_eigenval[:,id]=(eigenval_p-eigenval_m)/(2.0*dk)
          #
      end
      #
      # Convert from crystal to cartesian
      #
      dH_w       =K_crys_to_cart(dH_w,lattice)
      dU         =K_crys_to_cart(dU,lattice)
      dH_eigenval=K_crys_to_cart(dH_eigenval,lattice)
      #
    end
    #
    return dH_w,dU,dH_eigenval
end


function Evaluate_Dk_rho(rho, ik, k_grid, eigenvec, lattice)

  dk_rho=zeros(Complex{Float64},h_dim,h_dim,s_dim)
  for id in 1:s_dim
    #
    if k_grid.nk_dir[id]==1; continue end
    #
    ik_plus =get_k_neighbor(ik,id, 1,k_grid)
    ik_minus=get_k_neighbor(ik,id,-1,k_grid)
    #
    dk=norm(lattice.rvectors[id])/k_grid.nk_dir[id]
    #
    rho_plus =rho[:,:,ik_plus]
    rho_minus=rho[:,:,ik_minus]
    # 
    dk_rho[:,:,id]=(rho_plus-rho_minus)/(2.0*dk)
#    if ik==1
#	    println(rho[1,1,ik]," for ik ",ik)
#	    println(rho_plus[1,1]," for k-point ",ik_plus," and id ",id)
#	    println("k-point plus ",k_grid.kpt[:,ik_plus])
#	    println(rho_minus[1,1]," for k-point ",ik_minus," and id ",id)
#	    println("k-point minus ",k_grid.kpt[:,ik_minus])
#	    println(dk_rho[1,1,1])
#    end
    #
  end
  #
  # From crystal to cartesian coordinated
  #
  dk_rho=K_crys_to_cart(dk_rho,lattice)
  #
  return dk_rho
  #
end

function get_polarization(rho)
    pol=zeros(Float64,_dim)
    Threads.@threads for ik in 1:nk
        for id in 1:s_dim
	      if use_Dipoles
       	        pol[it,id]=pol[it,id]+real.(sum(Dip_h[:,:,ik,id].*transpose(rho_s[it,:,:,ik])))
	      else
      	        pol[it,id]=pol[it,id]+real.(sum(A_h[:,:,id,ik].*transpose(rho_s[it,:,:,ik])))
	      end
	   end
    end
    return pol
end
