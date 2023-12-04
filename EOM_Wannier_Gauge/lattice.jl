module LatticeTools

using LinearAlgebra

export Lattice,set_Lattice

mutable struct Lattice
    dim::Int8
    vectors::Array{Array{Float64,1},1}
    rvectors::Array{Array{Float64,1},1} #reciplocal lattice vectors
    vol::Float64
    r_vol::Float64
end


"""
    set_Lattice(dim::Integer,vectors::Array{Array{Float64,1},1})
Initialize lattice.
We have to call this before making Hamiltonian.
dim: Dimension of the system.
vector:: Primitive vectors.

Example:

1D system

```julia
la1 = set_Lattice(1,[[1.0]])
```

2D system

```julia
a1 = [sqrt(3)/2,1/2]
a2 = [0,1]
la2 = set_Lattice(2,[a1,a2])
```

3D system

```julia
a1 = [1,0,0]
a2 = [0,1,0]
a3 = [0,0,1]
la2 = set_Lattice(3,[a1,a2,a3])
```
"""
function set_Lattice(dim, vectors)
    #making primitive vectors
    pvector_1 = zeros(Float64, 3)
    pvector_2 = zeros(Float64, 3)
    pvector_3 = zeros(Float64, 3)
    if dim == 1
        pvector_1[1] = vectors[1][1]
        pvector_2[2] = 1.0 # 0 1 0
        pvector_3[3] = 1.0 # 0 0 1
    elseif dim == 2
        pvector_1[1:2] = vectors[1][1:2]
        pvector_2[1:2] = vectors[2][1:2]
        pvector_3[3] = 1.0 # 0 0 1
    elseif dim == 3
        pvector_1[1:3] = vectors[1][1:3]
        pvector_2[1:3] = vectors[2][1:3]
        pvector_3[1:3] = vectors[3][1:3]
    end
    #making reciplocal lattice vectors
    vol = dot(pvector_1, cross(pvector_2, pvector_3))
    rvector_1 = 2π * cross(pvector_2, pvector_3) / vol
    rvector_2 = 2π * cross(pvector_3, pvector_1) / vol
    rvector_3 = 2π * cross(pvector_1, pvector_2) / vol

    if vol< 0.0
       println("Axis vectors are left handed")
       vol=abs(vol)
    end
    
    if dim == 1
        rvectors = [[rvector_1[1]]]
	
    elseif dim == 2
        rvectors = [rvector_1[1:2], rvector_2[1:2]]
    elseif dim == 3
        rvectors = [rvector_1[1:3], rvector_2[1:3], rvector_3[1:3]]
    end

    r_vol=(2π)^3/vol

    println("Lattice vectors : ")
    for id in 1:dim
	    println(vectors[id][1:dim])
    end
    println("Reciprocal lattice vectors : ")
    for id in 1:dim
       println(rvectors[id][1:dim])
    end

    println("Direct lattice volume     : ",vol, " [a.u.]")
    println("Reciprocal lattice volume : ",r_vol, " [a.u.]")

    lattice = Lattice(
        dim,
	vectors,
	rvectors,
	vol,
	r_vol
    )
    return lattice
end

end 
