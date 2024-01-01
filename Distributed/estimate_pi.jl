using Distributed, Statistics

@everywhere function estimatepi(n)
 count = 0
 for i=1:n
   x = rand()
   y = rand()
   count += (x^2 + y^2) <= 1
 end
 return 4*count/n
end

parallelpi(N) = mean(pmap(n->estimatepi(n),[N/nworkers() for i=1:nworkers()]));
np = nprocs()
nw = nworkers()
println("number of processes : $np")
println("number of workers : $nw")

timestart = time()
# estpi = estimatepi(10_000_000_000)
estpi = parallelpi(10_000_000_000)
elapsed = time() - timestart
println("The estimate for Pi : $estpi")
println("The elapsed time : $elapsed seconds")
