#
# Density matrix EOM in the Wannier Gauge (TB approximation)
# Claudio Attaccalite (2023)
#
using LinearAlgebra
using Plots
using CSV
using DataFrames


include("TB_hBN.jl")
using .hBN2D

include("TB_tools.jl")
using .TB_tools

# This code is in Hamiltonian space
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

H_w=Hamiltonian(k_vec)
data= eigen(H_w)      # Diagonalize the matrix
eigenval = data.values
eigenvec = data.vectors

# Dipoles
Dip_w=Grad_H(k_vec)

# rotate in the Hamiltonian guage
Dip_h=zeros(Complex{Float64},h_dim,h_dim,s_dim)
for d in 1:s_dim
	Dip_h[:,:,d]=HW_rotate(Dip_w[:,:,d],eigenvec,"W_to_H")
# I set to zero the diagonal part of dipoles
	Dip_h[:,:,d]=Dip_h[:,:,d].*off_diag
end

# Now I have to dived for the energies
#
#  p = \grad_k H 
#
#  r_{ij} = i * p_{ij}/(e_j - e_i)
#
#  (diagonal terms are set to zero)
#
for i in 1:h_dim
    for j in i+1:h_dim
        Dip_h[i,j,:]=1im*Dip_h[i,j,:]/(eigenval[j]-eigenval[i])
        Dip_h[j,i,:]=conj(Dip_h[i,j,:])
    end
end

H_h=zeros(Complex{Float64},h_dim,h_dim)
for i in 1:h_dim
	H_h[i,i]=eigenval[i]
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

function deriv_rho(rho, t)::Vector{Complex{Float64}}
	#
	# Hamiltonian term
	#
        rho_mat=reshape(rho,h_dim,h_dim)
	d_rho=H_h*rho_mat-rho_mat*H_h
	#
	# Electrinc field
	#
	E_field=get_Efield(t, itstart=itstart)
	#
        E_dot_DIP=zeros(Complex{Float64},2,2)
        for d in 1:s_dim
            E_dot_DIP=E_dot_DIP-E_field[d]*Dip_h[:,:,d]
        end
        #
        # Commutator D*rho-rho*D
        #
        d_rho=d_rho-(E_dot_DIP*rho_mat-rho_mat*E_dot_DIP)
	#
	# Damping
	#
        damping=false
	if T_2!=0.0 && damping==true
		d_rho=d_rho-1im/T_2*off_diag.*rho_mat
	end
        d_rho_vec=-1.0im*reshape(d_rho,h_dim*h_dim)
        return d_rho_vec
end

function get_polarization(rho_solution)
    nsteps=size(rho_solution,1)
    pol=zeros(Float64,nsteps,s_dim)
    for it in 1:nsteps,id in 1:s_dim
        rho_t=view(rho_solution,it,:,:)
        pol[it,id]=real.(sum(Dip_h[:,:,id] .* rho_t ))
    end
    return pol
end 

rho0=zeros(Complex{Float64},h_dim,h_dim)
rho0[1,1]=1.0

# Solve EOM

solution = rungekutta2(deriv_rho, reshape(rho0,h_dim*h_dim), t_range)
solution_mat=reshape(solution,length(t_range),h_dim,h_dim)

# Calculate the polarization in time and frequency

pol=get_polarization(solution_mat)

# Write polarization and external field on disk

t_and_E=zeros(Float64,n_steps,3)
for it in 1:n_steps
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

