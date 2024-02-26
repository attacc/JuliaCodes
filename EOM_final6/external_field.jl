module ExternalField

export get_Efield

mutable struct Field_Properties
        itstart     ::Int
        Eint        ::Float64
        sigma       ::Float64
        E_vec       ::Array{Float64}
        ftype       ::String
end
        
e_field=(3,1.0,0.0,[0.0,1.0],"delta")

function get_Efield(t, e_field)
	#
	# Field in direction y
        # 
	T_0  = e_field.itstart*dt
        a_t=0.0
	#
        if lowercase(e_field.ftype)=="delta"
  	  #
          if t>=(e_field.itstart-1)*dt && t<e_field.itstart*dt 
  		a_t=1.0/dt
	  end
          #
       elseif lowertype(e_field.ftype)=="phhg"
	  if (t-T_0)>=sigma || (t-T_0)<0.0
	         a_t=0.0
	  else
                 a_t =(sin(pi*(t-T_0)/sigma))^2*sin(w*(t-T_0))
	  end
        elseif lowertype(e_field.ftype)=="sin"
          if t>=T_0; a_t=sin(w*(t-T_0)) end
	else
	  println("Field unknown!! ")	
	  exit()
	end
	#
	return a_t*Eamp
end
