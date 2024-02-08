#
# Density matrix EOM in the Wannier Gauge (TB approximation)
# Claudio Attaccalite (2023)
#
using LinearAlgebra
using Base.Threads

include("TB_hBN.jl")
using .hBN2D

include("TB_tools.jl")
using .TB_tools

include("units.jl")
using .Units

include("lattice.jl")
using .LatticeTools

include("bz_sampling.jl")
using .BZ_sampling

lattice=set_Lattice(2,[a_1,a_2])

# 
# Code This code is in Hamiltonian space

# a generic off-diagonal matrix example (0 1; 1 0)
off_diag=.~I(h_dim)

# K-points
n_k1=16
n_k2=16

k_grid=generate_unif_grid(n_k1, n_k2, lattice)
# print_k_grid(k_grid)

TB_sol.h_dim=h_dim # Hamiltonian dimension
TB_sol.eigenval=zeros(Float64,h_dim,k_grid.nk)
TB_sol.eigenvec=zeros(Complex{Float64},h_dim,h_dim,k_grid.nk)

println(" K-point list ")
println(" nk = ",k_grid.nk)
# for ik in 1:nk
#  	println(k_grid[:,ik])
# end
#
# Use only k=K
# k_grid[:,1]=b_mat*[1.0/3.0,-1.0/3.0]
#
# Select the space of the dynamics:
#
# Hamiltonian gauge:  h_gauge = true  
# Wannier gauge    :  h_gauge = false
#
dyn_props.h_gauge=false
#
# Add damping to the dynamics -i T_2 * \rho_{ij}
#
dyn_props.damping=true
#
# Use dipole d_k = d_H/d_k (in the Wannier guage)
#
dyn_props.use_dipoles=false
#
# Use UdU for dipoles
#
dyn_props.use_UdU_for_dipoles=true

# Include drho/dk in the dynamics
dyn_props.include_drho_dk=true
# Include A_w in the calculation of A_h
dyn_props.include_A_w=true #false

# Print properties on disk
props.print_dm =false
props.eval_curr=false
props.eval_pol =true

field_name="PHHG"
EInt = 2.64E8*kWCMm22AU

field_name="delta"
EInt  = 2.64E1*kWCMm22AU

Eamp =sqrt(EInt*4.0*pi/SPEED_of_LIGHT)

if dyn_props.h_gauge     
	println("* * * Hamiltonian gauge * * * ")             
else 
	println("* * * Wannier gauge * * * ") 
end
if dyn_props.use_dipoles 
	println("* * * Dipole approximation * * * ") 
else 
   println("* * * Full coupling with r = id/dk + A_w * * * ") 
end

if dyn_props.use_UdU_for_dipoles
   println("* * * Using UdU to build dipoles * * * ") 
else
   println("* * * Using dH/dk to build dipoles * * * ") 
end
if dyn_props.include_drho_dk
   println("* * * Using drho/dk in the dynamics * * * ") 
end

println(" Field name : ",field_name)
println(" Number of threads: ",Threads.nthreads())

#TB_gauge=TB_lattice
TB_gauge=TB_atomic
dk=nothing #0.01
println("Tight-binding gauge : $TB_gauge ")
println("Delta-k for derivatives : $dk ")

println("Orbitals coordinates : ")
println(orbitals.tau[1])
println(orbitals.tau[2])

println("Building Hamiltonian: ")
H_h=zeros(Complex{Float64},h_dim,h_dim,k_grid.nk)
TB_sol.H_w=zeros(Complex{Float64},h_dim,h_dim,k_grid.nk)

Threads.@threads for ik in ProgressBar(1:k_grid.nk)
    H_w=Hamiltonian(k_grid.kpt[:,ik],TB_gauge)
    data= eigen(H_w)      # Diagonalize the matrix
    TB_sol.eigenval[:,ik]   = data.values
    TB_sol.eigenvec[:,:,ik] = data.vectors
    if dyn_props.h_gauge
       for i in 1:h_dim
          H_h[i,i,ik]=TB_sol.eigenval[i,ik]
        end
    else
        H_h[:,:,ik]=H_w
    end
    TB_sol.H_w[:,:,ik]=H_w
    #
    # Fix phase of eigenvectors
    #
    TB_sol.eigenvec[:,:,ik]=fix_eigenvec_phase(TB_sol.eigenvec[:,:,ik])
    #
end

#Hamiltonian info
dir_gap=minimum(TB_sol.eigenval[2,:]-TB_sol.eigenval[1,:])
println("Direct gap : ",dir_gap*ha2ev," [eV] ")
ind_gap=minimum(TB_sol.eigenval[2,:])-maximum(TB_sol.eigenval[1,:])
println("Indirect gap : ",ind_gap*ha2ev," [eV] ")

# k-gradients of Hmailtonian, eigenvalues and eigenvectors
Dip_h=zeros(Complex{Float64},h_dim,h_dim,s_dim,k_grid.nk)
∇H_w =zeros(Complex{Float64},h_dim,h_dim,s_dim,k_grid.nk)

println("Building ∇H and Dipoles: ")
Threads.@threads for ik in ProgressBar(1:k_grid.nk)
# Dipoles 
  #Gradient of the Hamiltonian by finite difference 
  ∇H_w[:,:,:,ik]=Grad_H(ik, k_grid, lattice, Hamiltonian, TB_sol, TB_gauge,dk)
  #
  # Build dipoles
  for id in 1:s_dim
        Dip_h[:,:,id,ik]=HW_rotate(∇H_w[:,:,id,ik],TB_sol.eigenvec[:,:,ik],"W_to_H")
# I set to zero the diagonal part of dipoles
	Dip_h[:,:,id,ik]=Dip_h[:,:,id,ik].*off_diag
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
          Dip_h[i,j,:,ik]= 1im*Dip_h[i,j,:,ik]/(TB_sol.eigenval[j,ik]-TB_sol.eigenval[i,ik])
          Dip_h[j,i,:,ik]=conj(Dip_h[i,j,:,ik])
      end
  end
  #
  # I bring them back in Wannier gauge
  #
  if !dyn_props.h_gauge
    for id in 1:s_dim
        Dip_h[:,:,id,ik]=HW_rotate(Dip_h[:,:,id,ik],TB_sol.eigenvec[:,:,ik],"H_to_W")
     end
  end
end
#
# Calculate Overlaps in the Wannier Gauge
#
# Calculate A^H(k) 
#
println("Calculate A^H(k) : ")
A_h=zeros(Complex{Float64},h_dim,h_dim,s_dim,k_grid.nk)
A_w=zeros(Complex{Float64},h_dim,h_dim,s_dim,k_grid.nk)

if dyn_props.include_A_w
  println("Calculate A_w: ")
  Threads.@threads for ik in ProgressBar(1:k_grid.nk)
  #
  #
  # Calculate A(W) and rotate in H-gauge
  # Eq. II.13 of https://arxiv.org/pdf/1904.00283.pdf 
  # Notice that in TB-approxamation the Berry Connection is independent from k
  A_w[:,:,:,ik]=Berry_Connection(k_grid)
  #
  # Then I rotate from W -> H
  #
  for id in 1:s_dim
     A_h[:,:,id,ik]=HW_rotate(A_w[:,:,id,ik],TB_sol.eigenvec[:,:,ik],"W_to_H")
  end
  #
  end
end


println("Calculate UdU:")
Threads.@threads for ik in ProgressBar(1:k_grid.nk)
  #  
  # Calculate U^+ \d/dk U
  #
  #Gradient of eigenvectors/eigenvalus on the regular grid
  #
  UdU=Grad_U(ik, k_grid, lattice, TB_sol, TB_gauge,dk)
  #
  #  Using the fixing of the guage
  #
  U=TB_sol.eigenvec[:,:,ik]
  for id in 1:s_dim
    UdU[:,:,id]=(U')*UdU[:,:,id]
  end
  #
  A_h[:,:,:,ik]+=1im*UdU[:,:,:]
  #
  if dyn_props.use_UdU_for_dipoles
    Dip_h[:,:,:,ik]=1im*UdU[:,:,:]
    if !dyn_props.h_gauge
      for id in 1:s_dim
          Dip_h[:,:,id,ik]=HW_rotate(Dip_h[:,:,id,ik],TB_sol.eigenvec[:,:,ik],"H_to_W")
       end
    end
  end
  #
end
#
# Input paramters for linear optics with delta function
#
T_2=6.0*fs2aut   # fs
t_start=0.0
dt =0.01*fs2aut  # fs
t_end  =T_2*12.0
E_vec=[1.0,1.0]
#
t_range = t_start:dt:t_end
n_steps=size(t_range)[1]
#
# Buildo rho_0
#
rho0=zeros(Complex{Float64},h_dim,h_dim,k_grid.nk)
rho0[1,1,:].=1.0
#
# Transform in Wannier Gauge
#
if !dyn_props.h_gauge
  Threads.@threads for ik in 1:k_grid.nk
     rho0[:,:,ik]=HW_rotate(rho0[:,:,ik],TB_sol.eigenvec[:,:,ik],"H_to_W")
  end
end
#
#
#
println(" * * * Linear response from density matrix EOM within dipole approx. * * *")
println("Time rage ",t_start/fs2aut," - ",t_end/fs2aut)
println("Number of steps ",n_steps)
if dyn_props.damping
   println("Dephasing time ",T_2/fs2aut," [fs] ")
end
println("External field versor :",E_vec)

#
itstart = 10 # start of the external field

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
          a_t=a_t*Eamp
	  #
 	elseif ftype=="PHHG"
	  w    =0.77/ha2ev
	  sigma=30.0*fs2aut
	  T_0  = itstart*dt
	  if (t-T_0)>=sigma || (t-T_0)<0.0
	          a_t=0.0
	  else
		  a_t =Eamp*(sin(pi*(t-T_0)/sigma))^2*cos(w*t)
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
	d_rho=zeros(Complex{Float64},h_dim,h_dim,k_grid.nk) #
	#
	# Hamiltonian term
	#
        if dyn_props.h_gauge
          # in h-space the Hamiltonian is diagonal
	  Threads.@threads for ik in 1:k_grid.nk
             for ib in 1:h_dim, ic in 1:h_dim
  		   d_rho[ib,ic,ik] =TB_sol.eigenval[ib,ik]*rho[ib,ic,ik]-rho[ib,ic,ik]*TB_sol.eigenval[ic,ik]
             end
	  end
        else
          # in the w-space the Hamiltonian is not diagonal
	  Threads.@threads for ik in 1:k_grid.nk
 	     d_rho[:,:,ik]=TB_sol.H_w[:,:,ik]*rho[:,:,ik]-rho[:,:,ik]*TB_sol.H_w[:,:,ik]
	  end
        end
	#
	# Electrinc field
	#
	E_field=get_Efield(t, field_name,itstart=itstart)
	#
	Threads.@threads for ik in 1:k_grid.nk
	  #
	  # Dipole dot field
	  #
          E_dot_DIP=zeros(Complex{Float64},h_dim,h_dim)
	  #
          if dyn_props.use_dipoles
            for id in 1:s_dim
                E_dot_DIP+=E_field[id]*Dip_h[:,:,id,ik]
             end
          else
   	    if dyn_props.h_gauge
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
          if !dyn_props.use_dipoles
             # 
             # Add d_rho/dk
             #
	     if dyn_props.include_drho_dk
               Dk_rho=Evaluate_Dk_rho(rho, ik, k_grid, TB_sol.eigenvec, lattice)
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
	if T_2!=0.0 && dyn_props.damping==true
	   Threads.@threads for ik in 1:k_grid.nk
	     if dyn_props.h_gauge
	       d_rho[:,:,ik]=d_rho[:,:,ik]-1im/T_2*off_diag.*(rho[:,:,ik]-rho0[:,:,ik])
       	     else
	       rho_s=rho[:,:,ik]-rho0[:,:,ik]
	       rho_s=off_diag.*HW_rotate(rho_s,TB_sol.eigenvec[:,:,ik],"W_to_H")
	       damp_mat=HW_rotate(rho_s,TB_sol.eigenvec[:,:,ik],"H_to_W")
	       d_rho[:,:,ik]=d_rho[:,:,ik]-1im/T_2*damp_mat
	     end
	   end
	end

	d_rho.=-1im*d_rho
	
    return d_rho
end

function get_polarization(rho_s)
    nsteps=size(rho_s,1)
    pol=zeros(Float64,nsteps,s_dim)
    println("Calculate polarization: ")
    Threads.@threads for it in ProgressBar(1:nsteps)
      for ik in 1:k_grid.nk
         if !dyn_props.h_gauge
           rho=HW_rotate(rho_s[it,:,:,ik],TB_sol.eigenvec[:,:,ik],"W_to_H")
         else
           rho=rho_s[it,:,:,ik]
         end
         for id in 1:s_dim
            if !dyn_props.h_gauge
               dip=HW_rotate(Dip_h[:,:,id,ik],TB_sol.eigenvec[:,:,ik],"W_to_H").*off_diag
             else
               dip=Dip_h[:,:,id,ik].*off_diag
             end
       	     pol[it,id]=pol[it,id]+real.(sum(dip.*transpose(rho)))
	   end
        end
    end
    pol=pol/k_grid.nk
    return pol
end 

function get_current(rho_s)
    nsteps=size(rho_s,1)
    j_intra=zeros(Float64,nsteps,s_dim)
    j_inter=zeros(Float64,nsteps,s_dim)
    println("Calculate current: ")
    Threads.@threads for it in ProgressBar(1:nsteps)
      if dyn_props.h_gauge
        for ik in 1:k_grid.nk
	   rho_t_H=transpose(rho_s[it,:,:,ik])
           for id in 1:s_dim
     	      j_intra[it,id]=j_intra[it,id]-real.(sum(∇H_h[:,:,id,ik].*rho_t_H))
	      commutator=A_h[:,:,id,ik]*H_h[:,:,ik]-H_h[:,:,ik]*A_h[:,:,id,ik]
	      j_inter[it,id]=j_inter[it,id]-imag(sum(commutator.*rho_t_H))
	   end
	end
      else
        for ik in 1:k_grid.nk
	  rho_t_w=transpose(rho_s[it,:,:,ik])
           for id in 1:s_dim
     	      j_intra[it,id]=j_intra[it,id]-real.(sum(∇H_w[:,:,id,ik].*rho_t_w))
	      commutator=A_w[:,:,id,ik]*TB_sol.H_w[:,:,ik]-TB_sol.H_w[:,:,ik]*A_w[:,:,id,ik]
	      j_inter[it,id]=j_inter[it,id]-imag(sum(commutator.*rho_t_w))
	   end
	end
      end
    end
    j_intra=j_intra/k_grid.nk
    j_inter=j_inter/k_grid.nk
    return j_intra,j_inter
end 



# Solve EOM

solution = rungekutta2_dm(deriv_rho, rho0, t_range)


if props.eval_pol
  # Calculate the polarization in time and frequency
  pol=get_polarization(solution)
end
if props.eval_curr
  j_intra,j_inter=get_current(solution)
end

# Generate headers

function generate_header(k_grid=nothing,dyn_props=nothing,props=nothing)
   header="#\n# * * * EOM of the density matrix * * * \n#\n#"

   if k_grid != nothing
     header*="# k-point grid: $(k_grid.nk_dir[1]) - $(k_grid.nk_dir[2]) \n"
   end
   if dyn_props != nothing
     if dyn_props.h_gauge
         header*="# structure gauge : hamiltonian \n"
     else
         header*="# structure gauge : wannier \n"
     end
     header*="# include drho/dk in the dynamics: $(dyn_props.include_drho_dk) \n"
     header*="# include A_w in the dynamics: $(dyn_props.include_A_w) \n"
     header*="# use dipoles : $(dyn_props.use_dipoles) \n"
     header*="# use damping : $(dyn_props.damping) \n"
     header*="# use UdU for dipoles : $(dyn_props.use_UdU_for_dipoles) \n"
   end

   if props != nothing
     header*="# calculate polarization : $(props.eval_pol)\n"
     header*="# calculate current      : $(props.eval_curr)\n"
     header*="# print dm               : $(props.print_dm)\n"
   end
   header*="#\n#\n"
  return header
end


# Write polarization and external field on disk

t_and_E=zeros(Float64,n_steps,3)
Threads.@threads for it in 1:n_steps
 t=it*dt
 E_field_t=get_Efield(t,field_name,itstart=itstart)
 t_and_E[it,:]=[t/fs2aut,E_field_t[1],E_field_t[2]]
end

header=generate_header(k_grid,dyn_props,props)

if props.eval_pol
  f = open("polarization.csv","w") 
  write(f,header)
  write(f,"#time[fs] polarization_x  polarization_y\n") 
  for it in 1:n_steps
    write(f,"$(t_and_E[it,1]), $(pol[it,1]),  $(pol[it,2]) \n")
  end
  close(f)
end


if props.eval_curr
  f = open("current.csv","w") 
  write(f,header)
  write(f,"#time[fs] j_intra_x  j_intra_y  j_inter_x  j_inter_y\n") 
  for it in 1:n_steps
      write(f,"$(t_and_E[it,1]), $(j_intra[it,1]),  $(j_intra[it,2]),  $(j_inter[it,1]),  $(j_inter[it,2]) \n")
  end
  close(f)
end

f = open("external_field.csv","w") 
write(f,header)
write(f,"#time[fs]  efield_x efield_y\n") 
for it in 1:n_steps
  write(f,"$(t_and_E[it,1]),  $(t_and_E[it,2]),  $(t_and_E[it,3]) \n")
end

