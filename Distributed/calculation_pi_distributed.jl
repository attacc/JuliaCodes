using Distributed
using Random
using DistributedArrays
using Hwloc

println("Number of virtual  cores: $(num_virtual_cores()) ")
println("Number of physical cores: $(num_physical_cores()) ")

# Aggiungi lavoratori remoti (uno per ogni core)
addprocs(4)  # Imposta il numero di processi remoti desiderato

@everywhere function calculate_pi_distributed(number_of_points)
    points_inside = 0
    
    for _ in 1:number_of_points
        x, y = rand(), rand()
        distance = sqrt(x^2 + y^2)
        
        if distance <= 1.0
            points_inside += 1
        end
    end
    
    return points_inside
end

# Set the number of points for the simulation
number_of_points = 10000000000
println("Number of workers : $(nworkers())" )

# Calcola il valore di π utilizzando il calcolo distribuito
points_inside_total = sum(remotecall_fetch(calculate_pi_distributed, p, number_of_points ÷ nworkers()) for p in workers())

calculated_pi = 4 * points_inside_total / number_of_points

println("Estimated value of π with $number_of_points points: $calculated_pi")
