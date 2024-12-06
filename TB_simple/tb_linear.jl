#
# Simple Tight-binding code for one-dimensional system
# Claudio Attaccalite (2024)
#
using Printf
using LinearAlgebra
using PyPlot
using Bessels
using ArgParse
using TB_parms


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "-t"
            help = "diagonalize tb-Hamiltonian"
            action = :store_true
        "-f"
            help = "diagonalize flq-Hamiltonian"
            action = :store_true
        "-r"
            help = "real-time dynamics"
            action = :store_true
        "--Q"
            help = "gap parameter"
            arg_type = Float64
        "--nkpt"
            help = "number of k-ponints"
            arg_type = Int
        "--nmax"
            help = "max number of modes"
            arg_type = Int
        "--F"
            help = "Field intensity"
            arg_type = Float64
        "--omega"
            help = "Field frequency"
            arg_type = Float64
        "--tstep"
            help = "Time-step for real-time dynamics"
            arg_type = Float64
        "--nsteps"
            help = "Number of steps for real-time dynamics"
            arg_type = Int
    end

    return parse_args(s)
end

# In this program the lattice constant is equal to 1

function Hamiltonian(k,Q;At=0.0)::Matrix{Complex{Float64}}
        #
	H=zeros(Complex{Float64},2,2)
        #
        # Diagonal part 0,E_gap
        #
	H[1,1]=-Q
	H[2,2]=Q
        #
        # Off diagonal part
        #
        H[1,2]=cos(k[1]+At)
	H[2,1]=conj(H[1,2])
	return H
end

function Floquet_Hamiltonian(k, F_modes;  Q=0.0, omega=1.0, F=0.0)
        h_size =2
        n_modes=length(F_modes)
        H_flq=zeros(Complex{Float64},n_modes,n_modes,h_size,h_size)

#Diagonal terms respect to the mode and Q
        for i1 in 1:n_modes
          i_m=F_modes[i1]
          for ih in (1:h_size)
            H_flq[i1,i1,ih,ih]=i_m*omega+Q*(-1.0)^ih
          end
        end
        
#Off-diagonal terms in mode 
        for i1 in 1:n_modes
          i_m=F_modes[i1]
          for i2 in i1:n_modes
             i_n=F_modes[i2]
             H_flq[i1,i2,1,2]=(1.0im)^(i_m-i_n)*besselj(i_m-i_n,F)*cos(k[1]-(i_m-i_n)*pi/2.0)
             H_flq[i1,i2,2,1]=H_flq[i1,i2,1,2]
             if i1 != i2
                 H_flq[i2,i1,:,:]=conj(H_flq[i1,i2,:,:])
             end
          end
       end
 #My own reshape
       return copy(reshape(permutedims(H_flq,(1,3,2,4)),(n_modes*h_size,n_modes*h_size)))
end


    
#
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


function main()
    parsed_args = parse_commandline()
    #
    # generate k-points list
    # 
    n_kpt=parsed_args["nkpt"] 
    zero=[0.0]
    Pi2=[+pi/(2.0)]
    kpoints=[zero,Pi2]
    path=generate_circuit(kpoints,n_kpt)
    #    
    #tb-parameters
    #
    Q=parsed_args["Q"]
    if parsed_args["t"]
        TB_diag(path,Q)
    end
    #
    #
    # Floquet Hamiltonian parameters
    #
    F       =parsed_args["F"]    # Intensity
    omega   =parsed_args["omega"] # Frequency
    max_mode=parsed_args["nmax"]  # max number of modes
    if parsed_args["f"]
        FLQ_diag(path,Q,omega,F,max_mode)
    end
    #
    # Real-time
    #
    if parsed_args["r"]
       # RT_dynamics()
    end
end

function TB_diag(path,Q)
  println("")
  println(" * * * Tight-binding code for 1D-system  * * *")
  println("")

  band_structure = zeros(Float64, length(path), 2)
  
  for (i,kpt) in enumerate(path)
  	H=Hamiltonian(kpt,Q)
  	eigenvalues = eigen(H).values       # Diagonalize the matrix
        band_structure[i, :] = eigenvalues  # Store eigenvalues in an array
  end
  plot(band_structure[:, 1], label="Band 1")
  plot(band_structure[:, 2], label="Band 2")
  title("Band structure for two site 1D model")
  PyPlot.show()
end


function FLQ_diag(path,Q,omega,F,max_mode)
  h_size=2    # Hamiltonian size
  #
  F_modes=range(-max_mode,max_mode,step=1)
  n_modes=length(F_modes)
  #
  println("")
  @printf("Floquet Hamiltonian Q=%f  F=%f  omega=%f max_mode=%d ",Q,F,omega,max_mode)
  println("")
  #
  flq_bands = zeros(Float64, length(path), n_modes, h_size)
  for (i,kpt) in enumerate(path)
  	H_flq=Floquet_Hamiltonian(kpt,F_modes;Q,omega,F)
  	eigenvalues = eigen(H_flq).values       # Diagonalize the matrix
        flq_bands[i, :,:] = reshape(eigenvalues,(n_modes,h_size))  # Store eigenvalues in an array
  end
  plot(flq_bands[:, 1,1], label="Mode 1 band 1",color="green")
  plot(flq_bands[:, 1,2], label="Mode 1 band 2",color="green")
  plot(flq_bands[:, 2,1], label="Mode 2 band 1",color="blue")
  plot(flq_bands[:, 2,2], label="Mode 2 band 2",color="blue")
  plot(flq_bands[:, 3,1], label="Mode 3 band 1",color="red")
  plot(flq_bands[:, 3,2], label="Mode 3 band 2",color="red")
  title("Floquet band structure for two site 1D model")
  PyPlot.show()
end

function RT_dynamics(kpoints,Q,omega,F,tstep,nsteps)
  h_size=2    # Hamiltonian size
  #
  nk=length(kpoints)
  psi0=zeros(Complex{Float64},nk,h_size)
  #
  # Initial WF in the ground-state
  psi0[:,1]=1.0
  #
end

function time_evolution(ψ,time;t0=0.0)
        A_t=0.0
        if time>t0
           A_t=sim(omega(time-t0)


main()
