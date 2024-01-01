using Distributed

println(nprocs())
addprocs(4)         # add 4 workers
println(nprocs())   # total number of processes
println(nworkers()) # only worker processes
rmprocs(workers())  # remove worker processes
