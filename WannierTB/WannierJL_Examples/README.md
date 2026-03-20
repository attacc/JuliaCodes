# Wannier.jl Examples

This folder contains two Julia scripts that demonstrate how to construct and analyze Tight-Binding Hamiltonians directly using **Wannier.jl** (and `WannierIO`) instead of the custom parsing logic in `lumen/TightBinding`.

Wannier.jl is a powerful Julia library designed for post-processing Wannier90 inputs. It features native, highly optimized band structure interpolation, Berry phase algorithms, and full symmetry capability.

### Contents
1. **`1D_chain_example.jl`** 
   Demonstrates how to manually parse the `test_hr.dat` Real-Space matrices from the local testing directory and load them into a `Wannier.jl` `Hamiltonian` struct. It then evaluates the 1D band structure.

2. **`2D_boron_nitride_example.jl`**
   Loads the custom `boron_nitride_hr.dat` we constructed, builds the Hamiltonian in memory, and evaluates the precise bandgap splitting at the K-point (Dirac point) caused by the broken inversion symmetry in Boron Nitride.

### How to Run
Make sure `Wannier.jl` and `WannierIO.jl` are installed in your Julia environment. You can run these directly from your terminal while inside this project directory:

```bash
# Run the 1D Chain Example
julia --project=.. WannierJL_Examples/1D_chain_example.jl

# Run the 2D Boron Nitride Example
julia --project=.. WannierJL_Examples/2D_boron_nitride_example.jl
```

### Adding Symmetries
If you wish to apply spatial representation symmetries $D(\hat{S})_{ij}$ to these Hamiltonians to remove noise, `Wannier.jl` allows reading `.sym` files when you provide a full Wannier seed directory (i.e., using `read_w90("seedname")`). You can then cleanly apply structural symmetries natively to the loaded `Model`!
