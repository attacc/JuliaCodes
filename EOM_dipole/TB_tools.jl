module TB_tools

using Printf

export generate_circuit,generate_unif_grid,evaluate_DOS,rungekutta2
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
 #
 #
 function fermi_function(E, E_f, Temp)
   fermi_function=1.0/((exp(E-E_f)/Temp))
   return fermi_function
 end
 #
 function generate_unif_grid(n_kx, n_ky, b_1, b_2)
     k_grid=Any[]
     d_b1=b_1/n_kx
     d_b2=b_2/n_ky
     for ix in 0:(n_kx-1),iy in 0:(n_ky-1)
         vec=-b_1/2.0-b_2/2.0+d_b1*ix+d_b2*iy
         push!(k_grid,vec)
     end
     return k_grid
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

function rungekutta2(f, y0, t)
    n = length(t)
    y = similar(y0, n, length(y0))
    fill!(y, 0.0)
    y[1,:] = y0
    for i in 1:n-1
        h = t[i+1] - t[i]
        y[i+1,:] = y[i,:] + h * f(y[i,:] + f(y[i,:], t[i]) * h/2, t[i] + h/2)
    end
    return y
end
