
using FFTW
using Plots

t0 = 0              # Start time 
fs = 44100          # Sampling rate (Hz)
tmax = 1.0          # End time       

t = t0:1/fs:tmax;   
l = 0.02
signal = exp.((1im*2π* 60 - 1.0/l) .* t)
#signal = sin.(2π* 60 .* t)

F = fftshift(fft(signal))
println(" Fs ",fs)
println(" Length(t) ", length(t))
freqs = fftshift(fftfreq(length(t), fs))

# plots 
time_domain = plot(t, real.(signal), title = "Signal", label='f',legend=:top)
freq_domain = plot(freqs, real.(F), title = "Spectrum", xlim=(0, +200), xticks=0:20:200, label="abs.(F)",legend=:top) 
display(plot(time_domain, freq_domain, layout = (2,1)))
sleep(10)
