using Plots

b = 0.25
c = 5.0
y0 = [pi - 0.1; 0.0]

function f_pend(y, t)::Vector{Float64}
    return [y[2], (-b * y[2]) - (c * sin(y[1]))]
end

function rungekutta2(f, y0, t)
    n = length(t)
    y = zeros((n, length(y0)))
    y[1,:] = y0
    for i in 1:n-1
        h = t[i+1] - t[i]
        y[i+1,:] = y[i,:] + h * f(y[i,:] + f(y[i,:], t[i]) * h/2, t[i] + h/2)
    end
    return y
end


t3 = LinRange(0, 10, 101);
sol3 = rungekutta2(f_pend, y0, t3);


display(plot(t3, sol3[:, 1], xaxis="Time t", title="Solution to the pendulum ODE with Runge-Kutta 2 (21 points)", label="\\theta (t)"))
display(plot!(t3, sol3[:, 2], label="\\omega (t)"))
sleep(10)


