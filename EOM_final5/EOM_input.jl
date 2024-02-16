#
# Input file for the Density matrix EOM 
# Claudio Attaccalite (2023)
#

include("units.jl")
using .Units

include("TB_hBN.jl")
using .hBN2D

include("lattice.jl")
using .LatticeTools

include("TB_tools.jl")
using .TB_tools

include("bz_sampling.jl")
using .BZ_sampling

lattice=set_Lattice(2,[a_1,a_2])

n_k1=6
n_k2=6
#
# Integrator
#
Integrator=RK2
#
# Select the space of the dynamics:
#
# Hamiltonian gauge:  dyn_gauge = H_gauge
# Wannier gauge    :  dyn_gauge = W_gauge
#
dyn_props.dyn_gauge=W_gauge
#
# Add damping to the dynamics -i T_2 * \rho_{ij}
#
dyn_props.damping=true
#
# Use dipole d_k = d_H/d_k (in the Wannier guage)
#
dyn_props.use_dipoles=true
#
# Use UdU for dipoles
#
dyn_props.use_UdU_for_dipoles=true

# Include drho/dk in the dynamics
dyn_props.include_drho_dk=true
# Include A_w in the calculation of A_h
dyn_props.include_A_w=true

# Print properties on disk
props.print_dm  =false
props.eval_curr =true
props.curr_gauge=H_gauge
props.eval_pol  =true

field_name="PHHG"
EInt = 2.64E8*kWCMm22AU

#field_name="delta"
#EInt  = 2.64E8*kWCMm22AU

E_vec=[0.0,1.0]

#
# Gauge for the tight-binding
#
#TB_gauge=TB_lattice
TB_gauge=TB_atomic

#
# Step for finite differences in k-space
#
#dk=nothing 
dk=0.01
        
T_2=6.0*fs2aut      #  dephasing fs
T_1=12.0*fs2aut      #  dephasing fs
dt =0.0025*fs2aut   #  time-step fs
t_end  =72.0*fs2aut #  simulation lenght T_2*12.0
itstart = 20 # start of the external field
