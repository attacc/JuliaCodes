#
# Density matrix EOM in the Wannier Gauge (TB approximation)
# Claudio Attaccalite (2023)
#
using LinearAlgebra
using PyPlot;

include("TB_hBN.jl")
using .hBN2D

include("TB_tools.jl")
using .TB_tools
#
nk=40
n_kx=nk
n_ky=nk
#
n_bands=2

Berry_Conn = zeros(Float64, n_kx,n_ky)

k_x= zeros(Float64, n_kx)
k_y= zeros(Float64, n_ky)

limit_x=[-3.0,3.0]
limit_y=[-3.0,3.0]

dx=(limit_x[2]-limit_x[1])/n_kx
dy=(limit_y[2]-limit_x[1])/n_kx


for ix in 1:n_kx
   k_x[ix]=limit_x[1]+dx*(ix-1)
    for iy in 1:n_ky
	k_y[iy]=limit_y[1]+dy*(iy-1)
	k_vec=[k_x[ix],k_y[iy]]
        H=Hamiltonian(k_vec)
        data= eigen(H)      # Diagonalize the matrix
	eigenval = data.values
	eigenvec = data.vectors
        #
	# Fix phase of eigenvectors
	#
	eigenvec=fix_eigenvec_phase(eigenvec)
	#
        # Calculate A(W) and rotate in H-gauge
        # Eq. II.13 of https://arxiv.org/pdf/1904.00283.pdf 
        #
        A=Berry_Connection(k_vec)
        #
        # Add the dipole part. Since I cannot calculate directly
        # the derivative of U_k respect to k, due to the phase problem etc..
        # I use the formula Eq. 24 of https://arxiv.org/pdf/cond-mat/0608257.pdf
        # namely the simple dipole.
        # (In principle I can use the dynamical Berry phase formulation for this term)
        #
#        dH=Dipole_Matrices(k_vec)
#        A =A + dH
        #
	# Calculate U^+ \d/dk U
        #
	UdU=Calculate_UdU(k_vec, eigenvec)
	#
	#
        A=A+UdU
        #
        rot_A_x=HW_rotate(A[:,:,1],eigenvec,"W_to_H")
        rot_A_y=HW_rotate(A[:,:,2],eigenvec,"W_to_H")
        #
	Berry_Conn[ix,iy]=sqrt(abs(rot_A_x[1,2])^2+abs(rot_A_y[1,2])^2)
    end
end

plot_surface(k_y,k_y,Berry_Conn)
PyPlot.show()
