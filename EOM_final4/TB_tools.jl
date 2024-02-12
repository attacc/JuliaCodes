module TB_tools

using ProgressBars

export evaluate_DOS,rungekutta2_dm,fix_eigenvec_phase,ProgressBar,K_crys_to_cart,props,IO_output,dyn_props,TB_sol,Grad_H,Grad_U,W_gauge,H_gauge,HW_rotate,WH_rotate

const W_gauge=true
const H_gauge=false

mutable struct Properties
	eval_curr   ::Bool
	eval_pol    ::Bool
	print_dm    ::Bool
        curr_gauge  ::Bool
end

mutable struct Dynamics_Properties
	dyn_gauge    :: Bool
	damping	     :: Bool
	use_dipoles  :: Bool
	include_drho_dk :: Bool
	include_A_w	:: Bool
        use_UdU_for_dipoles :: Bool
end

mutable struct TB_Solution
        h_dim::Int
	eigenval::Array{Float64,2}
	eigenvec::Array{Complex{Float64},3}
	H_w::Array{Complex{Float64},3}
	TB_Solution() = new()
end

TB_sol=TB_Solution()

dyn_props = Dynamics_Properties(true,true,false,true,true,false)

props = Properties(true, true, false, W_gauge)

mutable struct IO_Output
	dm_file  ::Union{IOStream, Nothing}
	IO_Output(dm_file=nothing) =new(dm_file)
end

IO_output=IO_Output()

 #
 function fermi_function(E, E_f, Temp)
   fermi_function=1.0/((exp(E-E_f)/Temp))
   return fermi_function
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

@inline function HW_rotate(M,eigenvec)
  return eigenvec*M*adjoint(eigenvec)
end 

@inline function WH_rotate(M,eigenvec)
  return adjoint(eigenvec)*M*eigenvec
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
function rungekutta4_dm(d_rho, rho_0, t)
    n     = length(t)
    nk    = size(rho_0)[3]
    h_dim = size(rho_0)[1]
    rho_t = zeros(Complex{Float64},n, h_dim, h_dim, nk)
    rho_t[1,:,:,:] = rho_0

    println("Real-time equation integration: ")
    for i in ProgressBar(1:n-1)
        h = t[i+1] - t[i]
	rho_t[i+1,:,:,:] = rho_t[i,:,:,:] + h * d_rho(rho_t[i,:,:,:] + d_rho(rho_t[i,:,:,:], t[i]) * h/2, t[i] + h/2)
    end
    return rho_t
end
#
#
# Fix phase of the eigenvectors in such a way
# to have a positive definite diagonal
#
function fix_eigenvec_phase(eigenvec,ik=nothing,k_grid=nothing)
#	println("Before phase fixing : ")
#	show(stdout,"text/plain",eigenvec)
	#
	# Rotation phase matrix
	#
	phase_m=zeros(Complex{Float64},2)
	phase_m[1]=exp(-1im*angle(eigenvec[1,1]))
        phase_m[2]=exp(-1im*angle(eigenvec[2,2]))
	#
	# New eigenvectors
	#
#        if ik!=nothing
#          k_xy=k_grid.ik_map_inv[:,ik]
#          dk=zeros(Float64,2)
#          dk[1]=2.0*pi*(k_xy[1]-1.0)/k_grid.nk_dir[1]
#          dk[2]=2.0*pi*(k_xy[2]-1.0)/k_grid.nk_dir[2]
#          phase_m[1]*=exp(-1im*(dk[1]+dk[2]))
#          phase_m[2]*=exp(-1im*(dk[1]+dk[2]))
#        end
        #
	eigenvec[:,1]=eigenvec[:,1]*phase_m[1]
	eigenvec[:,2]=eigenvec[:,2]*phase_m[2]
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
# In this subroutine I do not recalculate H because
# I fix the phase
#
function Grad_U(ik, k_grid, lattice, TB_sol, TB_gauge, deltaK=nothing)
    #
    # calculate dH/dk in the Wannier Gauge
    # derivatives are in cartesian coordinates
    #
    h_dim=TB_sol.h_dim    # hamiltonian dimension
    s_dim=lattice.dim     # space dimension
    #
    dU         =zeros(Complex{Float64},h_dim,h_dim,s_dim)
    #
    # Derivative by finite differences, 
    # recalculating the Hamiltonian
    #
    k_plus =copy(k_grid.kpt[:,ik])
    k_minus=copy(k_grid.kpt[:,ik])
    #
    for id in 1:s_dim
      #  
      # In lattice gauge U is not periodic 
      # I need to recalculate it beyond the BZ
      #
      if k_grid.nk_dir[id]==1 && deltaK==nothing; continue end
      #
      if TB_gauge==TB_lattice || deltaK!=nothing
        #
        k_plus =copy(k_grid.kpt[:,ik])
        k_minus=copy(k_grid.kpt[:,ik])
        #
        if deltaK==nothing
          vec_dk=lattice.rvectors[id]/k_grid.nk_dir[id]
        else
          vec_dk=lattice.rvectors[id]*deltaK
        end
        dk=norm(vec_dk)
        #
        k_plus =k_plus +vec_dk
        k_minus=k_minus-vec_dk
        #
        H_plus =Hamiltonian(k_plus,  TB_gauge)
        H_minus=Hamiltonian(k_minus, TB_gauge)
        #  
        data_p=eigen(H_plus)      # Diagonalize the matrix
        data_m=eigen(H_minus)      # Diagonalize the matrix
        eigenvec_m= fix_eigenvec_phase(data_m.vectors)
        eigenvec_p= fix_eigenvec_phase(data_p.vectors)
        eigenval_m= data_m.values
        eigenval_p= data_p.values
        #
      elseif TB_gauge==TB_atomic && deltaK==nothing
        #
        # In the atomic gauge U is periodic, I can use the values in the grid
        # but I need to add the part coming from the Bloch phase
        #
        ik_plus  =get_k_neighbor(ik,id, 1,k_grid)
        ik_minus =get_k_neighbor(ik,id,-1,k_grid)
        #
        dk=norm(lattice.rvectors[id])/k_grid.nk_dir[id] 
        #
        eigenval_p=TB_sol.eigenval[:,ik_plus]
        eigenvec_p=TB_sol.eigenvec[:,:,ik_plus]
        #
        eigenval_m=TB_sol.eigenval[:,ik_minus]
        eigenvec_m=TB_sol.eigenvec[:,:,ik_minus]
        #
      end
      #
      dU[:,:,id]=(eigenvec_p-eigenvec_m)/(2.0*dk)
      #
      k_plus =copy(k_grid.kpt[:,ik])
      k_minus=copy(k_grid.kpt[:,ik])
      #     
    end
    #
    # Convert from crystal to cartesian
    #
    dU         =K_crys_to_cart(dU,lattice)
    #
    # Correction in atomic gauge
    #
    if TB_gauge==TB_atomic
      VdV=zeros(Complex{Float64},h_dim,h_dim,s_dim)
      for ih in 1:h_dim
        VdV[ih,ih,:]=-1im*orbitals.tau[ih][:]
      end
      U=TB_sol.eigenvec[:,:,ik]
      for id in 1:s_dim
        dU[:,:,id]=dU[:,:,id]+VdV[:,:,id]*U
      end
    end
    #
    return dU
    #
end

# For the derivative of the Hamiltonian
# I recalcualte it because H(k+G)/=H(k)

function Grad_H(ik, k_grid, lattice, Hamiltonian, TB_sol, TB_gauge, deltaK=nothing)
    #
    # calculate dH/dk in the Wannier Gauge
    # derivatives are in cartesian coordinates
    #
    h_dim=TB_sol.h_dim    # hamiltonian dimension
    s_dim=lattice.dim     # space dimension
    #
    dH_w       =zeros(Complex{Float64},h_dim,h_dim,s_dim)
    #
    # Derivative by finite differences, 
    # recalculating the Hamiltonian
    #
    for id in 1:s_dim
      #
      if k_grid.nk_dir[id]==1 && deltaK==nothing; continue end
      #
      if TB_gauge==TB_lattice || deltaK!=nothing
        #  
        k_plus =copy(k_grid.kpt[:,ik])
        k_minus=copy(k_grid.kpt[:,ik])
        #
        if deltaK==nothing
          vec_dk=lattice.rvectors[id]/k_grid.nk_dir[id]
        else
          vec_dk=lattice.rvectors[id]*deltaK
        end
        dk=norm(vec_dk)
        #
        k_plus =k_plus +vec_dk
        k_minus=k_minus-vec_dk
        #
        H_plus =Hamiltonian(k_plus,  TB_gauge)
        H_minus=Hamiltonian(k_minus, TB_gauge)
        #
      elseif TB_gauge==TB_atomic && deltaK==nothing
        #
        # In the atomic gauge H is periodic I can use the value on the grid
        # but I need to add the part coming from the Bloch phase
        #
        ik_plus  =get_k_neighbor(ik,id, 1,k_grid)
        ik_minus =get_k_neighbor(ik,id,-1,k_grid)
        #
        dk=norm(lattice.rvectors[id])/k_grid.nk_dir[id]
        #
        H_plus =TB_sol.H_w[:,:,ik_plus]
        H_minus=TB_sol.H_w[:,:,ik_minus]
        #
      end
      #
      dH_w[:,:,id]=(H_plus-H_minus)/(2.0*dk)
      #
    end
    #
    dH_w  =K_crys_to_cart(dH_w,lattice)
    #
    # If gauge is "atomic" apply correction to ∇H
    #
    if TB_gauge==TB_atomic
       d_tau=orbitals.tau[2]-orbitals.tau[1]     
       k_dot_dtau=dot(k_grid.kpt[:,ik],d_tau)     
       for id in 1:s_dim
         dH_w[1,2,id]=exp(1im*k_dot_dtau)*(dH_w[1,2,id]+1im*d_tau[id]*TB_sol.H_w[1,2,ik])
         dH_w[2,1,id]=conj(dH_w[1,2,id])
       end
    end
    #
    return dH_w
    #
end


function Evaluate_Dk_rho(rho, ik, k_grid, U, lattice)

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
    rho_k    =rho[:,:,ik]
    #
    dk_rho[:,:,id]=(rho_plus-rho_minus)/(2.0*dk)
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
