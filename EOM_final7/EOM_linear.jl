#
# Density matrix EOM in the Wannier Gauge (TB approximation)
# Claudio Attaccalite (2023)
#
using LinearAlgebra
using Base.Threads


include("EOM_input.jl")
# 
# EOM of the density matrix in Hamiltonian or Wannier gauge
#
k_grid=generate_unif_grid(n_k1, n_k2, lattice)

# a generic off-diagonal matrix example (0 1; 1 0)
off_diag=.~I(h_dim)

TB_sol.h_dim=h_dim # Hamiltonian dimension
TB_sol.eigenval=zeros(Float64,h_dim,k_grid.nk)
TB_sol.eigenvec=zeros(Complex{Float64},h_dim,h_dim,k_grid.nk)

println(" K-point list ")
println(" nk = ",k_grid.nk)

Eamp =sqrt(EInt*4.0*pi/SPEED_of_LIGHT)  # Do I miss a fator 2 in the sqrt? 8\pi instead of 4\pi
println("Field amplitute $Eamp  a.u. ")
println("Field amplitute $(Eamp*EAMPAU2VM/10e6*100)  M/Cm ")

if dyn_props.dyn_gauge==H_gauge     
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
    for i in 1:h_dim
      H_h[i,i,ik]=TB_sol.eigenval[i,ik]
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
        Dip_h[:,:,id,ik]=WH_rotate(∇H_w[:,:,id,ik],TB_sol.eigenvec[:,:,ik])
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
  if dyn_props.dyn_gauge==W_gauge
    for id in 1:s_dim
        Dip_h[:,:,id,ik]=HW_rotate(Dip_h[:,:,id,ik],TB_sol.eigenvec[:,:,ik])
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
     A_h[:,:,id,ik]=WH_rotate(A_w[:,:,id,ik],TB_sol.eigenvec[:,:,ik])
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
    if dyn_props.dyn_gauge==W_gauge
      for id in 1:s_dim
          Dip_h[:,:,id,ik]=HW_rotate(Dip_h[:,:,id,ik],TB_sol.eigenvec[:,:,ik])
       end
    end
  end
  #
end
#
# Input paramters for linear optics with delta function
#
t_range = 0.0:dt:t_end
n_steps=size(t_range)[1]
#
# Buildo rho_0
#
rho0=zeros(Complex{Float64},h_dim,h_dim,k_grid.nk)
rho0[1,1,:].=1.0
#
# Transform in Wannier Gauge
#
if dyn_props.dyn_gauge==W_gauge
  Threads.@threads for ik in 1:k_grid.nk
     rho0[:,:,ik]=HW_rotate(rho0[:,:,ik],TB_sol.eigenvec[:,:,ik])
  end
end
#
#
#
println(" * * * Linear response from density matrix EOM within dipole approx. * * *")
println("Time rage ",0.0," - ",t_end/fs2aut)
println("Number of steps ",n_steps)
if dyn_props.damping
   println("Dephasing time ",T_2/fs2aut," [fs] ")
   println("Life-time      ",T_1/fs2aut," [fs] ")
end
println("External field versor :",E_vec)
#
itstart = 20 # start of the external field

if dt!=nothing
  println("* * * * * Real-time dynamics not compatible with dt!=nothing * * * * * * ")
end

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
	  w    =0.4132/ha2ev # Parameter from De Silva
	  sigma=34.2*fs2aut  # Parameter from De Silva
	  T_0  = itstart*dt
	  if (t-T_0)>=sigma || (t-T_0)<0.0
	          a_t=0.0
	  else
                 a_t =Eamp*(sin(pi*(t-T_0)/sigma))^2*sin(w*(t-T_0))
	  end
	else
	  println("Field unknown!! ")	
	  exit()
	end
	#
	return a_t
end

# 
# Different possible matrix for the coupling
# with the external field in the commutator
#
# Dip_h= simple dipoles (ok for linear response)  dH/dk / (e_i -e_j)
# A_h  = Eq. II.13 of arXiv:1904.00283v2  (Berry connection in Hamiltonian gauge)
# A_w  = Eq. II.9 with approximation III.1 of of arXiv:1904.00283v2 (Berry connection in Wannier gauge)
#
if dyn_props.use_dipoles
  DIP_matrix=Dip_h
else
  if dyn_props.dyn_gauge==H_gauge
    DIP_maitrx=A_h
  elseif dyn_props.dyn_gauge==W_gauge
    DIP_matrix=A_w
  end
end
E_dot_DIP=zeros(Complex{Float64},h_dim,h_dim,k_grid.nk)
for id in 1:s_dim, ik in 1:k_grid.nk
  E_dot_DIP[:,:,ik]+=E_vec[id]*DIP_matrix[:,:,id,ik]
end


function deriv_rho(rho_in, t)
	#
        # Variable of the dynamics
        #
	h_dim=2
	#
	d_rho=zeros(Complex{Float64},h_dim,h_dim,k_grid.nk) #
        #
	# Hamiltonian term
	#
        if dyn_props.dyn_gauge==H_gauge
          # in h-space the Hamiltonian is diagonal
	  Threads.@threads for ik in 1:k_grid.nk
             for ib in 1:h_dim, ic in 1:h_dim
  	       d_rho[ib,ic,ik]=TB_sol.eigenval[ib,ik]*rho_in[ib,ic,ik]-rho_in[ib,ic,ik]*TB_sol.eigenval[ic,ik]
             end
	  end
        else
          # in the w-space the Hamiltonian is not diagonal
	  Threads.@threads for ik in 1:k_grid.nk
 	     d_rho[:,:,ik].=TB_sol.H_w[:,:,ik]*rho_in[:,:,ik]-rho_in[:,:,ik]*TB_sol.H_w[:,:,ik]
	  end
        end
	#
	# Electrinc field
	#
	E_field=get_Efield(t, field_name,itstart=itstart)
	#
	Threads.@threads for ik in 1:k_grid.nk
	  #
	  # Dipole term for the commutator dot field D*rho-rho*D
          # 
          d_rho[:,:,ik]+=E_field*(@view(E_dot_DIP[:,:,ik])*@view(rho_in[:,:,ik]))
          d_rho[:,:,ik]-=E_field*(@view(rho_in[:,:,ik])   *@view(E_dot_DIP[:,:,ik]))
          #
	end
        #
        if !dyn_props.use_dipoles &&  dyn_props.include_drho_dk
          # 
          # Add d_rho/dk
          #
   	  Threads.@threads for ik in 1:k_grid.nk
            Dk_rho=Evaluate_Dk_rho(rho_in, ik, k_grid, TB_sol.eigenvec, lattice)
            for id in 1:s_dim
              d_rho[:,:,ik].=d_rho[:,:,ik]+1im*E_field[id]*Dk_rho[:,:,id]
            end
            Dk_rho=nothing
          end
          #
        end
        #
	# Damping
	#
        if (T_2!=0.0 || T_1!=0.0) && dyn_props.damping==true
           #
           damp_mat=zeros(Float64,h_dim,h_dim)
           if T_2!=0.0
               damp_mat=1.0/T_2*off_diag
           end
           if T_1!=0.0
               damp_mat+=1.0/T_1*I(h_dim)
           end
           #
	   if dyn_props.dyn_gauge==H_gauge
	     Threads.@threads for ik in 1:k_grid.nk
               Δrho=rho_in[:,:,ik]-rho0[:,:,ik]
	       d_rho[:,:,ik]=d_rho[:,:,ik]-1im*damp_mat.*Δrho
             end
       	   else
	     Threads.@threads for ik in 1:k_grid.nk
               Δrho=rho_in[:,:,ik]-rho0[:,:,ik]
	       damp_dot_Δrho=damp_mat.*WH_rotate(Δrho,TB_sol.eigenvec[:,:,ik])
	       damp_dot_Δrho=HW_rotate(damp_dot_Δrho,TB_sol.eigenvec[:,:,ik])
	       d_rho[:,:,ik]=d_rho[:,:,ik]-1im*damp_dot_Δrho
	     end
	   end
           damp_mat=nothing
	end
        #
	Threads.@threads for ik in 1:k_grid.nk
	  d_rho[:,:,ik].=-1im*d_rho[:,:,ik]
	end
	#
    return d_rho
end

function polarization(rho)
  pol_t=zeros(Float64,s_dim)
  for ik in 1:k_grid.nk
    if dyn_props.dyn_gauge==W_gauge
       rho_in=WH_rotate(rho[:,:,ik],TB_sol.eigenvec[:,:,ik])
    else
      rho_in=rho[:,:,ik]
    end
    for id in 1:s_dim
       if dyn_props.dyn_gauge==W_gauge
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

function current(rho)
 j_intra_t=zeros(Float64, s_dim, Threads.nthreads())
 j_inter_t=zeros(Float64, s_dim, Threads.nthreads())
 if dyn_props.dyn_gauge==H_gauge
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
   if props.curr_gauge==H_gauge
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

   elseif props.curr_gauge==W_gauge
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
  j_intra=zeros(Float64,s_dim)
  j_inter=zeros(Float64,s_dim)
  for id in 1:s_dim
      j_intra[id]=sum(j_intra_t[id,:])/k_grid.nk
      j_inter[id]=sum(j_inter_t[id,:])/k_grid.nk
  end
  j_intra_t=nothing
  j_inter_t=nothing
  return j_intra,j_inter
end 

nsteps = length(t_range)

header=generate_header(k_grid,dyn_props,props)

if props.eval_curr
  j0_intra,j0_inter=current(rho0)
  f_curr = open("current.csv","w") 
  write(f_curr,header)
  write(f_curr,"#time[fs] j_intra_x  j_intra_y  j_inter_x  j_inter_y\n") 
end
if props.eval_pol
  f_pol = open("polarization.csv","w") 
  write(f_pol,header)
  write(f_pol,"#time[fs] polarization_x  polarization_y\n") 
end

f_field = open("external_field.csv","w") 
write(f_field,header)
write(f_field,"#time[fs]  efield_x efield_y\n") 

GC.gc()
println("Real-time equation integration: ")


let rho=copy(rho0)
  for it in ProgressBar(1:nsteps-1)
     #
     # Integration
     #
     if Integrator==RK2
      rk2_step!(rho, deriv_rho, t_range, it) 
    elseif Integrator==RK4
      rk4_step!(rho ,deriv_rho, t_range, it) 
    else
      println("Unknown integrator")
      exit()
    end
    #
    # Evaluate properties
    #
    if props.eval_curr;  
       j_intra,j_inter=current(rho)      
       j_intra-= j0_intra
       j_inter-= j0_inter
       write(f_curr,"$(t_range[it]/fs2aut), $(j_intra[1]),  $(j_intra[2]),  $(j_inter[1]),  $(j_inter[2]) \n")
       j_intra,j_inter=nothing,nothing
    end
    if props.eval_pol;   
       pol=polarization(rho) 
       write(f_pol,"$(t_range[it]/fs2aut), $(pol[1]),  $(pol[2]) \n")
       pol=nothing
    end    
    #
    # Write field on disk disk
    #
    E_field_t=get_Efield(t_range[it],field_name,itstart=itstart)
    write(f_field,"$(t_range[it]/fs2aut),  $(E_field_t*E_vec[1]),  $(E_field_t*E_vec[2]) \n")
    #
    if mod(it,1000)==0; GC.gc() end
    #
  end
end

close(f_field)
if props.eval_pol; close(f_pol) end
if props.eval_curr; close(f_curr) end
