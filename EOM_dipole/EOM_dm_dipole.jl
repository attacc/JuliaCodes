#
# Density matrix EOM in the Wannier Gauge (TB approximation)
# Claudio Attaccalite (2023)
#
using LinearAlgebra
using Plots
using CSV
using Tables


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
Dip_w=Dipole_Matrices(k_vec)

# rotate in the Hamiltonian guage
Dip_h=zeros(Complex{Float64},h_dim,h_dim,s_dim)
for d in 1:s_dim
	Dip_h[:,:,d]=HW_rotate(Dip_w[:,:,d],eigenvec,"W_to_H")
# I set to zero the diagonal part of dipoles
	Dip_h[:,:,d]=Dip_h[:,:,d].*off_diag
end

H_h=zeros(Complex{Float64},h_dim,h_dim)
for i in 1:h_dim
	H_h[i,i]=eigenval[i]
end

# 
# Input paramters for linear optics with delta function
#
T_2=5.0
t_start=0.0
dt =0.01
t_end  =T_2*8.0
E_vec=[1.0,0.0]
#
n_steps=floor(Integer,(t_end-t_start)/dt)

println(" * * * Linear response from density matrix EOM within dipole approx. * * *")
println("Time rage ",t_start,t_end)
println("Number of steps ",n_steps)
println("Dephasing time ",T_2)
println("External field versor ",E_vec)

#
t_range = t_start:dt:t_end

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
	E_field=get_Efield(t, itstart=50)
	#
        E_dot_DIP=zeros(Complex{Float64},2,2)
        for d in 1:s_dim
            E_dot_DIP=E_dot_DIP+E_field[d]*Dip_h[:,:,d]
        end
        #
        # Commutato D*rho-rho*D
        #
        d_rho=d_rho+1im*(E_dot_DIP*rho_mat-rho_mat*E_dot_DIP)
	#
	# Damping
	#
	if T_2!=0.0
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
        pol[it,id]=real.(1im*sum(Dip_h[:,:,id] .* rho_t' ))
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

# Write polarization on disk
header=["# Pol_x","Pol_y"] 
CSV.write("Polarization.csv", delim=" ", Tables.table(pol), header=header)


