#
# Density matrix EOM in the Wannier Gauge (TB approximation)
# Claudio Attaccalite (2023)
#
using LinearAlgebra
using CSV
using DataFrames
using Base.Threads

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

# a generic off-diagonal matrix example (0 1; 1 0)
off_diag=.~I(h_dim)

# K-points
n_k1=12
n_k2=12

k_grid=generate_unif_grid(n_k1, n_k2, lattice)
# print_k_grid(k_grid)

nk=n_k1*n_k2

eigenval=zeros(Float64,s_dim,nk)
eigenvec=zeros(Complex{Float64},s_dim,s_dim,nk)

println(" K-point list ")
println(" nk = ",nk)
# for ik in 1:nk
#  	println(k_grid[:,ik])
# end
#
# Use only k=K
# k_grid[:,1]=b_mat*[1.0/3.0,-1.0/3.0]
#
# Select the space of the dynamics:
#
# Hamiltonian gauge:  h_space = true  
# Wannier gauge    :  h_space = false
#
h_space=true
#
# Add damping to the dynamics -i T_2 * \rho_{ij}
#
damping=true
#
# Use dipole d_k = d_H/d_k (in the Wannier guage)
#
use_Dipoles=false
#use_Dipoles=false

# Include drho/dk in the dynamics
include_drho_dk=true

# Include A_w in the calculation of A_h
include_A_w=true

eval_current=true
eval_polarization=true

field_name="PHHG"

if h_space     println("* * * Hamiltonian gauge * * * ")             else println("* * * Wannier gauge * * * ") end
if use_Dipoles println("* * * Dipole approximation dk=dH/dk * * * ") else println("* * * Full coupling with r = id/dk + A_w * * * ") end
println(" Field name : ",field_name)
println(" Number of threads: ",Threads.nthreads())

println("Building Hamiltonian: ")
H_h=zeros(Complex{Float64},h_dim,h_dim,nk)
Threads.@threads for ik in ProgressBar(1:nk)
    H_w=Hamiltonian(k_grid.kpt[:,ik])
    data= eigen(H_w)      # Diagonalize the matrix
    eigenval[:,ik]   = data.values
    eigenvec[:,:,ik] = data.vectors
    if h_space
       for i in 1:h_dim
          H_h[i,i,ik]=eigenval[i,ik]
        end
    else
        H_h[:,:,ik]=H_w
    end
    #
    # Fix phase of eigenvectors
    #
    eigenvec[:,:,ik]=fix_eigenvec_phase(eigenvec[:,:,ik])
    #
end

#Hamiltonian info
dir_gap=minimum(eigenval[2,:]-eigenval[1,:])
println("Direct gap : ",dir_gap*ha2ev," [eV] ")
ind_gap=minimum(eigenval[2,:])-maximum(eigenval[1,:])
println("Indirect gap : ",ind_gap*ha2ev," [eV] ")

# rotate in the Hamiltonian guage
Dip_h=zeros(Complex{Float64},h_dim,h_dim,nk,s_dim)
Grad_h=zeros(Complex{Float64},h_dim,h_dim,nk,s_dim)

println("Building Dipoles: ")
Threads.@threads for ik in ProgressBar(1:nk)
# Dipoles
  #Gradient of the Hamiltonian along the cartesian directions
  Dip_w=Grad_H(k_grid.kpt[:,ik])

  for id in 1:s_dim
        Grad_h[:,:,ik,id]=HW_rotate(Dip_w[:,:,id],eigenvec[:,:,ik],"W_to_H")
# I set to zero the diagonal part of dipoles
	Grad_h[:,:,ik,id]=Grad_h[:,:,ik,id].*off_diag
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
          Dip_h[i,j,ik,:]= 1im*Grad_h[i,j,ik,:]/(eigenval[j,ik]-eigenval[i,ik])
          Dip_h[j,i,ik,:]=conj(Dip_h[i,j,ik,:])
      end
  end
  #
  # I bring them back in Wannier gauge
  #
  if !h_space
    for id in 1:s_dim
        Dip_h[:,:,ik,id]=HW_rotate(Dip_h[:,:,ik,id],eigenvec[:,:,ik],"H_to_W")
     end
  end
end
#
# Calculate Overlaps in the Wannier Gauge
#
# Calculate A^H(k) 
#
println("Calculate A^H(k) : ")
A_h=zeros(Complex{Float64},h_dim,h_dim,s_dim,nk)
A_w=zeros(Complex{Float64},h_dim,h_dim,s_dim,nk)
Threads.@threads for ik in ProgressBar(1:nk)
  #
  if include_A_w
    #
    # Calculate A(W) and rotate in H-gauge
    # Eq. II.13 of https://arxiv.org/pdf/1904.00283.pdf 
    # Notice that in TB-approxamation the Berry Connection is independent from k
    A_w[:,:,:,ik]=Berry_Connection(k_grid.kpt[:,ik])
    #
    # Calculate Berry Connection using Eq. 44 of PRB 74, 195118 (2006) 
#    A_w[:,:,:,ik]=Calculate_Berry_Conec_FD(ik, k_grid, eigenvec[:,:,ik])
    #
    # Then I rotate from W -> H
    #
    for id in 1:s_dim
       A_h[:,:,id,ik]=HW_rotate(A_w[:,:,id,ik],eigenvec[:,:,ik],"W_to_H")
    end
    #
    #
  end
  # Calculate U^+ \d/dk U
  #
  #  Using the fixing of the guage
#  UdU=Calculate_UdU(ik,k_grid, eigenvec, lattice)
#  UdU=UdU #.*off_diag
  #
  #  Using the parallel transport gauge
  UdU=zeros(Complex{Float64},h_dim,h_dim,s_dim)
  for id in 1:s_dim
     UdU[:,:,id]=-1im*Dip_h[:,:,ik,id]
  end
  #
  A_h[:,:,:,ik]=A_h[:,:,:,ik]+1im*UdU
  #
end
#
# 
# Input paramters for linear optics with delta function
#
T_2=6.0*fs2aut   # fs
t_start=0.0
dt =0.01*fs2aut  # fs
t_end  =T_2*12.0
E_vec=[1.0,0.0]
#
t_range = t_start:dt:t_end
n_steps=size(t_range)[1]
#
# Buildo rho_0
#
rho0=zeros(Complex{Float64},h_dim,h_dim,nk)
rho0[1,1,:].=1.0
#
# Transform in Wannier Gauge
#
if !h_space
  Threads.@threads for ik in 1:nk
     rho0[:,:,ik]=HW_rotate(rho0[:,:,ik],eigenvec[:,:,ik],"H_to_W")
  end
end
#
#
#
println(" * * * Linear response from density matrix EOM within dipole approx. * * *")
println("Time rage ",t_start/fs2aut," - ",t_end/fs2aut)
println("Number of steps ",n_steps)
if damping
   println("Dephasing time ",T_2/fs2aut," [fs] ")
end
println("External field versor :",E_vec)

#
itstart = 3 # start of the external field

function get_Efield(t, ftype; itstart=3)
	#
	# Field in direction y
	#
	if ftype=="delta"
  	  #
          if t>=(itstart-1)*dt && t<itstart*dt 
  		a_t=1.0/dt
  	  else
		a_t=0.0
	  end
	  #
 	elseif ftype=="PHHG"
	  w    =0.77/ha2ev
	  sigma=30.0*fs2aut
	  T_0  = itstart*dt
	  I    = 2.64E10*kWCMm22AU
          E    =sqrt(I*4.0*pi/SPEED_of_LIGHT)
	  if (t-T_0)>=sigma || (t-T_0)<-sigma
	    a_t=0.0
	  else
		  a_t =E*(sin(pi*(t-T_0)/sigma))^2*cos(w*t)
	  end
	else
	  println("Field unknown!! ")	
	  exit()
	end
	#
	Efield=a_t*E_vec
	#
	return Efield
end

function deriv_rho(rho, t)
	#
        # Variable of the dynamics
        #
	h_dim=2
        #
	#
	d_rho=zeros(Complex{Float64},h_dim,h_dim,nk) #
	#
	# Hamiltonian term
	#
        if h_space
          # in h-space the Hamiltonian is diagonal
	  Threads.@threads for ik in 1:nk
             for ib in 1:h_dim
                for ic in 1:h_dim
 	          d_rho[ib,ic,ik] =H_h[ib,ib,ik]*rho[ib,ic,ik]-rho[ib,ic,ik]*H_h[ic,ic,ik]
                end
             end
	  end
        else
          # in the w-space the Hamiltonian is not diagonal
	  Threads.@threads for ik in 1:nk
 	     d_rho[:,:,ik]=H_h[:,:,ik]*rho[:,:,ik]-rho[:,:,ik]*H_h[:,:,ik]
	  end
        end
	#
	# Electrinc field
	#
	E_field=get_Efield(t, field_name,itstart=itstart)
	#
	Threads.@threads for ik in 1:nk
	  #
	  # Dipole dot field
	  #
          E_dot_DIP=zeros(Complex{Float64},h_dim,h_dim)
	  #
          if use_Dipoles
            for id in 1:s_dim
                E_dot_DIP+=E_field[id]*Dip_h[:,:,ik,id]
             end
          else
   	    if h_space
              for id in 1:s_dim
                 E_dot_DIP+=E_field[id]*A_h[:,:,id,ik]
              end
	    else
              for id in 1:s_dim
                 E_dot_DIP+=E_field[id]*A_w[:,:,id,ik]
              end
	    end
          end
          #
          if !use_Dipoles
             # 
             # Add d_rho/dk
             #
	     if include_drho_dk
               Dk_rho=Evaluate_Dk_rho(rho, h_space, ik, k_grid, eigenvec, lattice)
               #
	       #
               for id in 1:s_dim
                 d_rho[:,:,ik]+=1im*E_field[id]*Dk_rho[:,:,id]
               end
	       #
	     end
             #
           end
           #
           # Commutator D*rho-rho*D
           # 
	   d_rho[:,:,ik]=d_rho[:,:,ik]+(E_dot_DIP[:,:]*rho[:,:,ik]-rho[:,:,ik]*E_dot_DIP[:,:])
           #
           # 
	end
	#
	# Damping
	#
	if T_2!=0.0 && damping==true
	   Threads.@threads for ik in 1:nk
	     if h_space
	       d_rho[:,:,ik]=d_rho[:,:,ik]-1im/T_2*off_diag.*(rho[:,:,ik]-rho0[:,:,ik])
       	     else
	       rho_s=rho[:,:,ik]-rho0[:,:,ik]
	       rho_s=off_diag.*HW_rotate(rho_s,eigenvec[:,:,ik],"W_to_H")
	       damp_mat=HW_rotate(rho_s,eigenvec[:,:,ik],"H_to_W")
	       d_rho[:,:,ik]=d_rho[:,:,ik]-1im/T_2*damp_mat
	     end
	   end
	end

	d_rho.=-1.0im*d_rho
	
    return d_rho
end

function get_polarization(rho_s)
    nsteps=size(rho_s,1)
    pol=zeros(Float64,nsteps,s_dim)
    println("Calculate polarization: ")
    Threads.@threads for it in ProgressBar(1:nsteps)
        for id in 1:s_dim
           for ik in 1:nk
	      if use_Dipoles
       	        pol[it,id]=pol[it,id]+real.(sum(Dip_h[:,:,ik,id].*transpose(rho_s[it,:,:,ik])))
	      else
      	        pol[it,id]=pol[it,id]+real.(sum(A_h[:,:,id,ik].*transpose(rho_s[it,:,:,ik])))
	      end
	   end
        end
    end
    pol=pol/nk
    return pol
end 

function get_current(rho_s)
    nsteps=size(rho_s,1)
    j_intra=zeros(Float64,nsteps,s_dim)
    j_inter=zeros(Float64,nsteps,s_dim)
    println("Calculate current: ")
    Threads.@threads for it in ProgressBar(1:nsteps)
      for ik in 1:nk
	if !h_space
	    rho_t_H=transpose(HW_rotate(rho_s[it,:,:,ik],eigenvec[:,:,ik],"W_to_H"))
	else
	    rho_t_H=transpose(rho_s[it,:,:,ik])
	end
        for id in 1:s_dim
     	   j_intra[it,id]=j_intra[it,id]-real.(sum(Grad_h[:,:,ik,id].*rho_t_H))
	      if h_space
	         commutator=A_h[:,:,id,ik]*H_h[:,:,ik]-H_h[:,:,ik]*A_h[:,:,id,ik]
	      else
	         commutator=A_w[:,:,id,ik]*H_h[:,:,ik]-H_h[:,:,ik]*A_w[:,:,id,ik]
	      end
	   j_inter[it,id]=j_inter[it,id]-imag(sum(commutator.*transpose(rho_s[it,:,:,ik])))
	end
      end
    end
    j_intra=j_intra/nk
    j_inter=j_inter/nk
    return j_intra,j_inter
end 

# Solve EOM

solution = rungekutta2_dm(deriv_rho, rho0, t_range)


if eval_polarization
  # Calculate the polarization in time and frequency
  pol=get_polarization(solution)
end
if eval_current
  j_intra,j_inter=get_current(solution)
end

  # Write polarization and external field on disk

t_and_E=zeros(Float64,n_steps,3)
Threads.@threads for it in 1:n_steps
 t=it*dt
 E_field_t=get_Efield(t,field_name,itstart=itstart)
 t_and_E[it,:]=[t/fs2aut,E_field_t[1],E_field_t[2]]
end

if eval_polarization
  df = DataFrame(time  = t_and_E[:,1], 
               pol_x = pol[:,1], 
               pol_y = pol[:,2],
               )
  f = open("polarization.csv","w") 
  CSV.write(f, df; quotechar=' ', delim=' ')
end


if eval_current
  df = DataFrame(time  = t_and_E[:,1], 
               j_intra_x = j_intra[:,1], 
               j_intra_y = j_intra[:,2],
               j_inter_x = j_inter[:,1], 
               j_inter_y = j_inter[:,2],
               )
  f = open("current.csv","w") 
  CSV.write(f, df; quotechar=' ', delim=' ')
end

header2=["time [fs]", "efield_x","efield_y"] 
CSV.write("external_field.csv", delim=' ', Tables.table(t_and_E), header=header2, quotechar=' ')

