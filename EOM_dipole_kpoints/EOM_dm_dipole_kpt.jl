#
# Density matrix EOM in the Wannier Gauge (TB approximation)
# Claudio Attaccalite (2023)
#
using LinearAlgebra
using Plots
using CSV
using DataFrames
using Base.Threads

include("TB_hBN.jl")
using .hBN2D

include("TB_tools.jl")
using .TB_tools
# 
# Code This code is in Hamiltonian space
# in dipole approximation only at the K-point
#

k_vec=[1,1/sqrt(3)]
k_vec=k_vec*2*pi/3

# Hamiltonian dimension
h_dim=2
# Space dimension
s_dim=2



# a generic off-diagonal matrix example (0 1; 1 0)
off_diag=.~I(h_dim)

# K-points
n_k1=96
n_k2=96

b_1=2*pi/3.0*[1.0, -sqrt(3)]
b_2=2*pi/3.0*[1.0,  sqrt(3)]

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

for ik in 1:nk
        H_w=Hamiltonian(k_list[:,ik])
	data= eigen(H_w)      # Diagonalize the matrix
	eigenval[:,ik]   = data.values
	eigenvec[:,:,ik] = data.vectors
end

# rotate in the Hamiltonian guage
Dip_h=zeros(Complex{Float64},h_dim,h_dim,nk,s_dim)

for ik in 1:nk
# Dipoles
  Dip_w=Grad_H(k_list[:,ik])
  for id in 1:s_dim
        Dip_h[:,:,ik,id]=HW_rotate(Dip_w[:,:,id],eigenvec[:,:,ik],"W_to_H")
# I set to zero the diagonal part of dipoles
	Dip_h[:,:,ik,id]=Dip_h[:,:,ik,id].*off_diag
  end
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
	for ik in 1:nk
          Dip_h[i,j,ik,:]= 1im*Dip_h[i,j,ik,:]/(eigenval[j,ik]-eigenval[i,ik])
          Dip_h[j,i,ik,:]=conj(Dip_h[i,j,ik,:])
	end
    end
end

H_h=zeros(Complex{Float64},h_dim,h_dim,nk)
for i in 1:h_dim
	H_h[i,i,:]=eigenval[i,:]
end

# 
# Input paramters for linear optics with delta function
#
T_2=6.0*fs2aut   # fs
t_start=0.0
dt =0.005*fs2aut  # fs
t_end  =T_2*10.0
E_vec=[1.0,0.0]
#
t_range = t_start:dt:t_end
n_steps=size(t_range)[1] 

println(" * * * Linear response from density matrix EOM within dipole approx. * * *")
println("Time rage ",t_start/fs2aut," - ",t_end/fs2aut)
println("Number of steps ",n_steps)
println("Dephasing time ",T_2/fs2aut," [fs] ")
println("External field versor :",E_vec)

#
itstart = 3 # start of the external field

function get_Efield(t ; itstart=3)
	#
	# Field in direction y
	#
        if t>=(itstart-1)*dt && t<itstart*dt 
		a_t=1.0/dt
	else
		a_t=0.0
	end
	#
	Efield=a_t*E_vec
	#
	return Efield
end

function deriv_rho(rho, t)
	#
	h_dim=2
	#
	d_rho=zeros(Complex{Float64},h_dim,h_dim,nk) #
	#
	# Hamiltonian term
	#
	Threads.@threads for ik in 1:nk
 	   d_rho[:,:,ik]=H_h[:,:,ik]*rho[:,:,ik]-rho[:,:,ik]*H_h[:,:,ik]
	end
	#
	# Electrinc field
	#
	E_field=get_Efield(t, itstart=itstart)

    E_dot_DIP=zeros(Complex{Float64},h_dim,h_dim)
	#
	Threads.@threads for ik in 1:nk
	   #
	   # Dipole dot field
	   #
	   E_dot_DIP.=0.0
	   #
        for id in 1:s_dim
			E_dot_DIP[:,:]=E_dot_DIP[:,:]-E_field[id]*Dip_h[:,:,ik,id]
        end
            # 
            # Commutator D*rho-rho*D
            # 
	    d_rho[:,:,ik]=d_rho[:,:,ik]-(E_dot_DIP[:,:]*rho[:,:,ik]-rho[:,:,ik]*E_dot_DIP[:,:])
	end
	#
	# Damping
	#
        damping=false
	if T_2!=0.0 && damping==true
		Threads.@threads for ik in 1:nk
	      d_rho[:,:,ik]=d_rho[:,:,ik]-1im/T_2*off_diag.*rho[:,:,ik]
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
     	    	pol[it,id]=pol[it,id]+real.(sum(Dip_h[:,:,ik,id] .* transpose(rho_s[it,:,:,ik])))
			end
		end
    end
    pol=pol/nk
    return pol
end 

rho0=zeros(Complex{Float64},h_dim,h_dim,nk)
rho0[1,1,:].=1.0

# Solve EOM

solution = rungekutta2_dm(deriv_rho, rho0, t_range)

# Calculate the polarization in time and frequency

pol=get_polarization(solution)

# Write polarization and external field on disk

t_and_E=zeros(Float64,n_steps,3)
Threads.@threads for it in 1:n_steps
 t=it*dt
 E_field_t=get_Efield(t,itstart=itstart)
 t_and_E[it,:]=[t/fs2aut,E_field_t[1],E_field_t[2]]
end


df = DataFrame(time  = t_and_E[:,1], 
               pol_x = pol[:,1], 
               pol_y = pol[:,2],
               )
f = open("polarization.csv","w") 
CSV.write(f, df; quotechar=' ', delim=' ')

header2=["time [fs]", "efield_x","efield_y"] 
CSV.write("external_field.csv", delim=' ', Tables.table(t_and_E), header=header2, quotechar=' ')

