function fft_pol(pol)
    pol_along_Efield=pol*E_vec
    F = fftshift(fft(pol_along_Efield))
    @show length(t_range),dt
    freqs = fftshift(fftfreq(length(t_range), dt))
    return F,freqs,pol_along_Efield
end

pol_w,freqs,pol_Edir=fft_pol(pol)

# Write data on file

FileIO.save("polarization.dat", "pol", pol)  
