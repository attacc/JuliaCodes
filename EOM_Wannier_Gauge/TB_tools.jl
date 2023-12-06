module TB_tools

using Printf
using ProgressBars

export generate_circuit,generate_unif_grid,evaluate_DOS,rungekutta2_dm,fix_eigenvec_phase,get_k_neighbor,ProgressBar

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
     k_grid=zeros(Float64,lattice.dim,nk)
     ik_grid    =zeros(Int,n_kx,n_ky)
     ik_grid_inv=zeros(Int,lattice.dim,nk)

     vec_crystal=zeros(Float64,lattice.dim)
     dx=1.0/n_kx
     dy=1.0/n_ky

     ik=1
     for ix in 0:(n_kx-1),iy in 0:(n_ky-1)
           ik_grid[ix+1,iy+1]=ik
           ik_grid_inv[:,ik]   =[ix+1,iy+1]
	   vec_crystal[1]=dx*ix
	   vec_crystal[2]=dy*iy
	   k_grid[:,ik]=lattice.rvectors[1]*vec_crystal[1]+lattice.rvectors[2]*vec_crystal[2]
	   ik=ik+1
     end
     return k_grid,ik_grid,ik_grid_inv
 end
 #
 function get_k_neighbor(ik,id,istep,ik_grid,ik_grid_inv)
     s_dim=2
     ik_n =0

     k_xyz   =ik_grid_inv[:,ik]
     k_n     =k_xyz
     k_n[id] =k_n[id] 
#
     for id in 1:s_dim
        d_size=size(ik_grid)[id]
        if k_n[id] > d_size
            k_n[id]=k_n[id]-d_size
        elseif k_n[id]<=0
            k_n[id]=k_n[id]+d_size
        end
     end
#
     if s_dim==1
             ik_n=ik_grid[k_n[1]]
     elseif s_dim ==2
             ik_n=ik_grid[k_n[1],k_n[2]]
     elseif s_dim ==3
             ik_n=ik_grid[k_n[1],k_n[2],k_n[3]]
     end

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

function rungekutta2_dm(d_rho, rho_0, t)
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

function Evaluate_Dk_rho(rho, ik, k_grid, ik_grid, ik_grid_inv, eigenvec, lattice)

  dk_rho=zeros(Complex{Float64},h_dim,h_dim,s_dim)
  for id in 1:s_dim
    #  
    ik_plus =get_k_neighbor(ik,id, 1,ik_grid,ik_grid_inv)
    ik_minus=get_k_neighbor(ik,id,-1,ik_grid,ik_grid_inv)
    #
    rho_plus =HW_rotate(rho[:,:,ik_plus],eigenvec[:,:,ik_plus],"H_to_W")
    rho_minus=HW_rotate(rho[:,:,ik_minus],eigenvec[:,:,ik_minus],"H_to_W")
    #
    dk=norm(k_grid[:,ik_plus]-k_grid[:,ik_minus])/2.0
    # 
    dk_rho[:,:,id]=(rho[:,:,ik_plus]-rho[:,:,ik_minus])/(2.0*dk)
    #
    dk_rho[:,:,id]=HW_rotate(dk_rho[:,:,id],eigenvec[:,:,ik],"W_to_H")
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

function BerryConnection_by_Finite_Diff(ik, k_grid, ik_grid, ik_grid_inv, eigenvec)
    #
    # We used the formula Eq. 44 of PRB 74 195118 to
    # calculate the BerryConnection by finite difference
    #
    BerryCon=zeros(Complex{Float64},h_dim,h_dim,s_dim)
    #
    return BerryConnection
end

