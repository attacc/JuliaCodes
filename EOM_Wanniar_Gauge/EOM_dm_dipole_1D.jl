#
# Density matrix EOM in the Wannier Gauge (TB approximation)
# Claudio Attaccalite (2023)
#
using LinearAlgebra
using Plots;

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

T_2=10.0

t_start=0.0
t_end  =10.0
n_steps=1000
dt     =(t_end-t_start)/(n_steps-1)
t_range = LinRange(t_start, t_end, n_steps);

function get_Efield(t)
	#
	# Field in direction y
	#
	if dt<=t && t<2*dt 
		a_t=1.0
	else
		a_t=0.0
	end
	#
	Efield=a_t
	#
	return Efield
end

function deriv_rho(rho, t)::Vector{Complex{Float64}}
	#
	# Hamiltonian term
	#
	d_rho=rho[1]*(H_h[1,1]-H_h[2,2])
	#
	# Electrinc field
	#
	E_field=get_Efield(t)
	#
	d_rho=d_rho+1im*E_field*Dip_h[1,2,1]
	#
	# Damping
	#
	if T_2!=0.0
		d_rho=d_rho+1im/T_2*d_rho
	end
	return [-1im*d_rho]
end

rho0=zeros(Complex{Float64},1)
solution = rungekutta2(deriv_rho, rho0, t_range);
display(plot(t_range, real.(solution[:, 1])))
sleep(10)



