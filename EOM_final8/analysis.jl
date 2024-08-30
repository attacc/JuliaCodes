#
# Analysis of the time dependent polarization
# Claudio Attaccalite (2023)
#
using CSV
using DataFrames
using FFTW
using ProgressBars
using PyPlot

include("EOM_input.jl")

# function fft_pol(time, pol,e_vec)
#     pol_along_Efield=pol*e_vec
#     pol_w = fftshift(fft(pol_along_Efield))
#     dt=time[2]-time[1]
#     n_steps=length(time)
#     freqs = fftshift(fftfreq(n_steps, dt))
#     return pol_w,freqs,pol_along_Efield
# end

function FFT_1D(times, freqs, pol, e_vec)
    pol_w=zeros(Complex{Float64},length(freqs))
    pol_edir=pol*e_vec
    dt=times[2]-times[1]
    println("FFT: ")
    for itime in ProgressBar(1:length(times))
      for (ifreqs, freq) in enumerate(freqs)
          pol_w[ifreqs] =pol_w[ifreqs]+exp(-1im*freq*times[itime])*pol_edir[itime]
      end
    end
    pol_w=pol_w*dt
end

function Divide_by_the_field(pol_w, times, itstart)
  #
  # I have to divide for the Fourier transform of the external field
  #
  # E(w) = \int dt E(t)  e^{ - omega * t }
  #
  # for a delta function
  #
  # E(w) = \int dt \delta(t-t_0} e^{-omega * t} = e^{ -omega t_0} 
  #
  tstart=times[itstart]
  println("Starting time of delta function: ",tstart/fs2aut)
  #
  for (ifreqs, freq) in enumerate(freqs)
     pol_w[ifreqs]=pol_w[ifreqs]*exp(1im * tstart*freq)
  end
  #
  Int  = 2.64E1*kWCMm22AU
  Eamp =sqrt(Int*4.0*pi/SPEED_of_LIGHT)
  pol_w.=pol_w/Eamp
  #
  return pol_w
  #
end

function damp_it(times, pol,T2,itstart)
    tstart=times[itstart]
    for (itime,time) in enumerate(times)
        if itime >= itstart
            damp_func=exp(-(time-tstart)/T2)
        else
            damp_func=1.0
        end
            pol[itime,:].=pol[itime,:]*damp_func
    end
    return pol
end

df = CSV.read("polarization.csv",DataFrame, comment="#",delim=',')
pol_and_times=Matrix(df)

s_dim=size(pol_and_times)[2]-1
n_steps=size(pol_and_times)[1]
println("Spacial dimensions :",s_dim)
println("Number of steps    :",n_steps)

pol  =pol_and_times[:,2:s_dim+1]
println(pol_and_times[1:10,1])
times=pol_and_times[:,1]*fs2aut

freqs_range  =[0.0/ha2ev, 25.0/ha2ev] # eV
freqs_nsteps =250

freqs=LinRange(freqs_range[1],freqs_range[2],freqs_nsteps)
#
# Additional damping in post-processing
#
if T2_PP !=0.0
   pol  =damp_it(times, pol, T2_PP, itstart)
end
pol_w=FFT_1D(times, freqs, pol, e_field.E_vec)
# I multiply for -1im lost somewhere
pol_w=-1im*Divide_by_the_field(pol_w,times,itstart)
#
# Write data on file
#
df_out= DataFrame( freqs = freqs*ha2ev, re_pol_w = real.(pol_w), im_pol_w=imag.(pol_w))
CSV.write("pol_w.cvs", df_out; quotechar=' ', delim=' ')  

fig = figure("Real-time response",figsize=(10,20))
plot(freqs*ha2ev,imag(pol_w))
plot(freqs*ha2ev,real(pol_w))
PyPlot.show();

