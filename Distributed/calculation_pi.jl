function calculate_pi(number_of_points)
    points_inside = 0
    
    for _ in 1:number_of_points
        x, y = rand(), rand()
        distance = sqrt(x^2 + y^2)
        
        if distance <= 1.0
            points_inside += 1
        end
    end
    
    return 4 * points_inside / number_of_points
end

# Set the number of points for the simulation
number_of_points = 1000000

# Calculate the value of π
calculated_pi = calculate_pi(number_of_points)

println("Estimated value of π with $number_of_points points: $calculated_pi")

