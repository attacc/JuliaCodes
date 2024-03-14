using HopTB
using PyPlot

lat = [1 1/2 0; 0 √3/2 0; 0 0 1]
site_positions = lat*([1/3 1/3 0; 2/3 2/3 0]');
# TBmodel the second vector is orbital type (s in this case one for site)
tm = TBModel(lat,site_positions,[[0], [0]])

# Parameters from Phys. Rev. B 94, 125303
t_0 =2.30 # eV
E_gap=3.625*2.0 # eV

addhopping!(tm, [0, 0, 0], (1, 1), (1, 1), -E_gap/2.0)
addhopping!(tm, [0, 0, 0], (2, 1), (2, 1), E_gap/2.0)

addhopping!(tm, [0, 0, 0], (1, 1), (2, 1), t_0)
addhopping!(tm, [-1, 0, 0], (1, 1), (2, 1), t_0)
addhopping!(tm, [0, -1, 0], (1, 1), (2, 1), t_0)

# Calculate optics
# α=1 (x direction)
# β=1 (x direction)
ω_range=collect(4:0.05:14)
χ = HopTB.Optics.getpermittivity(tm, 1, 1, ω_range, 0.0, [96, 96, 1], ϵ=0.15)

fig = figure("Optical response of monolayer hBN",figsize=(10,20))

plot(ω_range,real(χ))
plot(ω_range,imag(χ))
PyPlot.show()
