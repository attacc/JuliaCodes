#
# Analysis of the time dependent polarization
# Claudio Attaccalite (2023)
#
using CSV
using DataFrames
using FFTW

include("TB_hBN.jl")
using .hBN2D

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
    for (itime,time) in enumerate(times)
      for (ifreqs, freq) in enumerate(freqs)
          pol_w[ifreqs] =pol_w[ifreqs]+exp(-1im*freq*time)*pol_edir[itime]
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
  tstart=itstart*(times[2]-times[1])
  println("Starting time of delta function: ",tstart/fs2aut)
  #
  for (ifreqs, freq) in enumerate(freqs)
     pol_w[ifreqs]=pol_w[ifreqs]*exp(1im * tstart*freq)
  end
  #
  return pol_w
  #
end

df = CSV.read("polarization.csv",DataFrame)
pol=Matrix(df)
e_vec=[1.0, 0.0]

freqs_range  =[0.0/ha2ev, 20.0/ha2ev] # eV
freqs_nsteps =200

freqs=LinRange(freqs_range[1],freqs_range[2],freqs_nsteps)

pol_w=FFT_1D(pol[:,1]*fs2aut, freqs, pol[:,2:3], e_vec)
itstart=3 # should be the same of the EOM_dm_dipoles.jl
pol_w=Divide_by_the_field(pol_w,pol[:,1]*fs2aut,itstart)
#
# Write data on file
#
df_out= DataFrame( freqs = freqs*ha2ev, re_pol_w = real.(pol_w), im_pol_w=imag.(pol_w))
CSV.write("pol_w.cvs", df_out; quotechar=' ', delim=' ')  
