using HopTB.Parallel: parallel_sum, parallel_do
using Distributed

function pippo(
    ik::Integer
    )
    println("Ciao sono qui $ik")
    return 1.0*ik
end

batchsize=1
nsum=12

println("Number of workers $(nworkers()) ") 
println("Workers $(workers()) ") 

mutable struct ParallelFunction
    jobs::RemoteChannel
    results::RemoteChannel
    nworkers::Int64
    nreceived::Int64
    ntaken::Int64
    isstopped::Bool
end

function do_work(f, jobs, results, args...)
    while true
        x = take!(jobs)
        if isnothing(x) return nothing end
        result = try
            f(x, args...)
        catch err
            RemoteException(CapturedException(err, stacktrace(catch_backtrace())))
        end
        put!(results, result)
    end
end


function start_work(f, jobs, results, args...)
    @async do_work(f, jobs, results, args...)
    return nothing
end

function ParallelFunction(f::Function, args...; len=5*nworkers(), processes=workers())
    jobs = RemoteChannel(()->Channel{Any}(len))
    results = RemoteChannel(()->Channel{Any}(len))
    for p in processes
        remotecall_fetch(start_work, p, f, jobs, results, args...)
    end
    return ParallelFunction(jobs, results, length(processes), 0, 0, false)
end


pf = ParallelFunction(pippo)
#pf = parallel_doParallelFunction(pippo, a)
#result=parallel_sum(ik -> pippo(ik),1:nsum,0.0;batchsize=1)
#

#pf = ParallelFunction(get_value_k, len=nworkers())
#@async for item in 1:nsum
##   pf(item)
#end

result = 0.0

sum=0.0
for ik in 1:nsum
    global sum+=ik
end
println("Result parallel: $result ")
println("Result serial  : $sum ")

