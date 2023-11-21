#
# Analysis of the time dependent polarization
# Claudio Attaccalite (2023)
#
using CSV
using DataFrames
using FFTW

include("TB_hBN.jl")
using .hBN2D

function fft_pol(time, pol,e_vec)
    pol_along_Efield=pol*e_vec
    pol_w = fftshift(fft(pol_along_Efield))
    dt=time[2]-time[1]
    n_steps=length(time)
    freqs = fftshift(fftfreq(n_steps, dt))
    return pol_w,freqs,pol_along_Efield
end



df = CSV.read("polarization.csv",DataFrame)
pol=Matrix(df)
e_vec=[1.0, 0.0]

pol_w, freqs, pol_along_Efield=fft_pol(pol[:,1]*fs2aut,pol[:,2:3],e_vec)
#
freqs=freqs*ha2ev
#
# Write data on file
#
df_out= DataFrame( freqs = freqs, re_pol_w = real.(pol_w), im_pol_w=imag.(pol_w))
CSV.write("pol_w.cvs", df_out; quotechar=' ', delim=' ')  
