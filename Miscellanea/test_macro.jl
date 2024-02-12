macro commutator(A,B)
    @assert 
    quote
       A*B-B*A
    end
end

C=@show_value(1.0,2.0)
println("$C")
#@show_value(apple)
