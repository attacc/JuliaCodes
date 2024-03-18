module ExternalField

include("units.jl")
using .Units

export get_Efield,e_field

mutable struct Field_Properties
        itstart     ::Int
        EInt        ::Float64
        sigma       ::Float64
        w           ::Float64
        E_vec       ::Array{Float64}
        ftype       ::String
end
        
e_field=Field_Properties(3,1.0,0.0,0.0,[0.0,1.0],"delta")

function get_Efield(t, dt, e_field)
	#
	# Field in direction y
        # 
	T_0  = e_field.itstart*dt
        Eamp =sqrt(e_field.EInt*4.0*pi/SPEED_of_LIGHT)  
        # Do I miss a fator 2 in the sqrt? 8\pi instead of 4\pi
        a_t=0.0
	#
        if lowercase(e_field.ftype)=="delta"
  	  #
          if t>=(T_0-dt)*dt && t<T_0
  		a_t=1.0/dt
	  end
          #
       elseif lowercase(e_field.ftype)=="phhg"
	  if (t-T_0)>=e_field.sigma || (t-T_0)<0.0
	         a_t=0.0
	  else
                 a_t =(sin(pi*(t-T_0)/e_field.sigma))^2*sin(e_field.w*(t-T_0))
	  end
        elseif lowercase(e_field.ftype)=="sin"
          if t>=T_0; a_t=sin(e_field.w*(t-T_0)) end
	else
	  println("Field unknown!! ")	
	  exit()
	end
	#
	return a_t*Eamp
end
end
